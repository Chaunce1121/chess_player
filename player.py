import re
import random
import chess
import torch

from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament import Player

# Basic piece values used for simple capture scoring
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def piece_value(piece_type: Optional[int]) -> int:
    # Returns the value of a chess piece
    if piece_type is None:
        return 0
    return PIECE_VALUE.get(piece_type, 0)


def count_attackers(board: chess.Board, color: bool, square: chess.Square) -> int:
    # Counts how many pieces of a color attack a given square
    return len(board.attackers(color, square))


def king_safety_score(board: chess.Board, pov: bool) -> int:
    # Simple king safety bonus based on castling and king defenders/attackers
    score = 0
    my_king = board.king(pov)

    if my_king is not None:
        enemy_attackers = count_attackers(board, not pov, my_king)
        own_defenders = count_attackers(board, pov, my_king)
        score -= 18 * enemy_attackers
        score += 8 * own_defenders

        # Small bonus for castled king positions
        if pov == chess.WHITE and my_king in [chess.G1, chess.C1]:
            score += 35
        if pov == chess.BLACK and my_king in [chess.G8, chess.C8]:
            score += 35

    return score


def move_heuristic(board: chess.Board, move: chess.Move, pov: bool) -> int:
    # Scores a move using a few simple heuristics
    score = 0
    moving_piece = board.piece_at(move.from_square)
    target_piece = board.piece_at(move.to_square)

    # Play immediate mate if available
    b2 = board.copy()
    b2.push(move)
    if b2.is_checkmate():
        return 10**9

    # Reward promotions (pawns)
    if move.promotion:
        score += 800 if move.promotion == chess.QUEEN else 300

    # Reward castling because it usually improves king safety
    if board.is_castling(move):
        score += 100

    # Reward captures, especially valuable ones
    if board.is_capture(move):
        captured_value = piece_value(target_piece.piece_type) if target_piece else 100
        attacker_value = piece_value(moving_piece.piece_type) if moving_piece else 0
        score += 20 + captured_value - (attacker_value // 10)

    # Reward checks
    if board.gives_check(move):
        score += 40

    # Penalize landing on an attacked square without enough defense
    moved_piece = b2.piece_at(move.to_square)
    if moved_piece and moved_piece.color == pov and moved_piece.piece_type != chess.KING:
        enemy_attackers = count_attackers(b2, not pov, move.to_square)
        own_defenders = count_attackers(b2, pov, move.to_square)

        if enemy_attackers > 0 and own_defenders == 0:
            score -= piece_value(moved_piece.piece_type) // 2
        elif enemy_attackers > own_defenders:
            score -= piece_value(moved_piece.piece_type) // 4

    # Add a small king safety bonus after the move
    score += king_safety_score(b2, pov)

    return score


class TransformerPlayer(Player):
    # Custom chess player that combines a transformer model with simple heuristics
    def __init__(
        self,
        name: str,
        hf_model_id: str = "Chaunce1121/chess-fen-move-model",
        num_tries: int = 12,
    ):
        super().__init__(name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.num_tries = num_tries

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt(self, fen: str) -> str:
        # Creates the input prompt for the language model
        return f"FEN: {fen}\nMOVE:"

    def _extract(self, txt: str) -> List[str]:
        # Extracts UCI formatted moves from model output
        tail = txt.split("MOVE:", 1)[-1].lower()
        return re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", tail)

    def _generate_candidates(self, board: chess.Board, fen: str) -> List[chess.Move]:
        # Uses the transformer multiple times to generate legal candidate moves
        prompt = self._prompt(fen)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        candidates: List[chess.Move] = []

        for _ in range(self.num_tries):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=12,
                    do_sample=True,
                    top_k=80,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            txt = self.tokenizer.decode(out[0], skip_special_tokens=True)

            for uci in self._extract(txt):
                try:
                    mv = chess.Move.from_uci(uci)
                    if mv in board.legal_moves:
                        candidates.append(mv)
                except ValueError:
                    pass

        # Add some random legal moves if the model gives too few valid moves
        legal_moves = list(board.legal_moves)
        if len(candidates) < 6:
            extra = random.sample(legal_moves, k=min(10, len(legal_moves)))
            candidates.extend(extra)

        # Remove duplicates while keeping order
        seen = set()
        uniq = []
        for m in candidates:
            if m not in seen:
                seen.add(m)
                uniq.append(m)

        return uniq

    def get_move(self, fen: str) -> Optional[str]:
        # Main function that picks the final move
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        pov = board.turn

        # Simple preferred opening moves
        if board.fullmove_number == 1:
            preferred = ["e2e4", "d2d4", "g1f3", "c2c4"] if pov == chess.WHITE else ["e7e5", "d7d5", "g8f6", "c7c5"]
            for uci in preferred:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    return mv.uci()

        # Generate candidate moves using the transformer
        candidates = self._generate_candidates(board, fen)

        # Pick the move with the best simple heuristic score
        best_move = None
        best_score = -10**18

        for mv in candidates:
            score = move_heuristic(board, mv, pov)
            if score > best_score:
                best_score = score
                best_move = mv

        if best_move:
            return best_move.uci()

        return random.choice(legal_moves).uci()
