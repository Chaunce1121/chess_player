import re
import random
import chess
import torch

from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament import Player

# Piece value when captured
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Important center squares that are generally strong to control
CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

def piece_value(piece_type: Optional[int]) -> int:
    # Returns the value of a chess piece
    if piece_type is None:
        return 0
    return PIECE_VALUE.get(piece_type, 0)


def material_score(board: chess.Board, pov: bool) -> int:
    # Calculates material advantage between both players
    score = 0
    for ptype, val in PIECE_VALUE.items():
        score += val * len(board.pieces(ptype, pov))
        score -= val * len(board.pieces(ptype, not pov))
    return score

def count_attackers(board: chess.Board, color: bool, square: chess.Square) -> int:
    # Counts how many pieces of a color attack a given square
    return len(board.attackers(color, square))

def development_score(board: chess.Board, pov: bool) -> int:
    # Rewards developing knights and bishops from their starting squares
    score = 0

    # Reward development of knights and bishops off the back rank in opening/middlegame
    if pov == chess.WHITE:
        undeveloped = {
            chess.B1: chess.KNIGHT,
            chess.G1: chess.KNIGHT,
            chess.C1: chess.BISHOP,
            chess.F1: chess.BISHOP,
        }
    else:
        undeveloped = {
            chess.B8: chess.KNIGHT,
            chess.G8: chess.KNIGHT,
            chess.C8: chess.BISHOP,
            chess.F8: chess.BISHOP,
        }

    for sq, ptype in undeveloped.items():
        piece = board.piece_at(sq)
        if not (piece and piece.color == pov and piece.piece_type == ptype):
            score += 12

    return score


def center_control_score(board: chess.Board, pov: bool) -> int:
    # Gives a bonus for occupying or attacking central squares
    score = 0

    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == pov:
            score += 16
        elif piece and piece.color != pov:
            score -= 16

        score += 4 * count_attackers(board, pov, sq)
        score -= 4 * count_attackers(board, not pov, sq)

    for sq in EXTENDED_CENTER:
        score += 1 * count_attackers(board, pov, sq)
        score -= 1 * count_attackers(board, not pov, sq)

    return score

def king_safety_score(board: chess.Board, pov: bool) -> int:
    # Evaluates how safe the king is based on attackers and defenders
    score = 0

    my_king = board.king(pov)
    opp_king = board.king(not pov)

    if my_king is not None:
        attackers = count_attackers(board, not pov, my_king)
        defenders = count_attackers(board, pov, my_king)
        score -= 18 * attackers
        score += 8 * defenders

    if opp_king is not None:
        attackers = count_attackers(board, pov, opp_king)
        defenders = count_attackers(board, not pov, opp_king)
        score += 14 * attackers
        score -= 6 * defenders

    # Reward castled king positions a bit
    if pov == chess.WHITE:
        if my_king in [chess.G1, chess.C1]:
            score += 35
    else:
        if my_king in [chess.G8, chess.C8]:
            score += 35

    return score


def hanging_pieces_penalty(board: chess.Board, pov: bool) -> int:
    # Penalizes pieces that are attacked but not defended
    penalty = 0

    for square, piece in board.piece_map().items():
        if piece.color != pov:
            continue
        if piece.piece_type == chess.KING:
            continue

        enemy_attackers = count_attackers(board, not pov, square)
        own_defenders = count_attackers(board, pov, square)

        if enemy_attackers > 0 and own_defenders == 0:
            penalty += piece_value(piece.piece_type) // 3
        elif enemy_attackers > own_defenders:
            penalty += piece_value(piece.piece_type) // 5

    return penalty


def mobility_score(board: chess.Board, pov: bool) -> int:
    # Rewards positions where the player has more legal moves
    current_turn = board.turn

    board.turn = pov
    my_mobility = board.legal_moves.count()

    board.turn = not pov
    opp_mobility = board.legal_moves.count()

    board.turn = current_turn
    return 2 * (my_mobility - opp_mobility)


def evaluate_board(board: chess.Board, pov: bool) -> int:
    # Combines all evaluation components into a final board score
    score = 0
    score += material_score(board, pov)
    score += development_score(board, pov)
    score += center_control_score(board, pov)
    score += king_safety_score(board, pov)
    score += mobility_score(board, pov)
    score -= hanging_pieces_penalty(board, pov)

    if board.is_check():
        if board.turn != pov:
            # opponent to move and in check
            score += 25
        else:
            # we are to move and in check
            score -= 25

    return score


def move_heuristic(board: chess.Board, move: chess.Move, pov: bool) -> int:
    # Quickly evaluates a move based on captures, checks, promotions, etc.
    score = 0
    moving_piece = board.piece_at(move.from_square)
    target_piece = board.piece_at(move.to_square)

    # Big bonus for pawn promotion
    if move.promotion:
        score += 800 if move.promotion == chess.QUEEN else 350

    # Castling
    if board.is_castling(move):
        score += 120

    # Reward capturing opponent pieces
    if board.is_capture(move):
        captured_value = piece_value(target_piece.piece_type) if target_piece else 100
        attacker_value = piece_value(moving_piece.piece_type) if moving_piece else 0
        score += 20 + captured_value - (attacker_value // 8)

    # Bonus for putting the opponent in check
    if board.gives_check(move):
        score += 45

    # Encourage moving pieces toward the center
    if move.to_square in CENTER_SQUARES:
        score += 20
    elif move.to_square in EXTENDED_CENTER:
        score += 8

    # Opening development
    if board.fullmove_number <= 10 and moving_piece:
        if moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            score += 18
        if moving_piece.piece_type == chess.QUEEN:
            score -= 12  # discourage early queen adventure

    # Penalize moving into attacked square if not well defended
    b2 = board.copy()
    b2.push(move)

    moved_piece = b2.piece_at(move.to_square)
    if moved_piece and moved_piece.color == pov and moved_piece.piece_type != chess.KING:
        # Penalize moves where the piece lands on a dangerous square
        enemy_attackers = count_attackers(b2, not pov, move.to_square)
        own_defenders = count_attackers(b2, pov, move.to_square)

        if enemy_attackers > 0 and own_defenders == 0:
            score -= piece_value(moved_piece.piece_type) // 2
        elif enemy_attackers > own_defenders:
            score -= piece_value(moved_piece.piece_type) // 4

    return score


class TransformerPlayer(Player):
    # Loads the HuggingFace transformer model used to generate moves
    def __init__(
        self,
        name: str,
        hf_model_id: str = "Chaunce1121/chess-fen-move-model",
        num_tries: int = 12
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
        # Uses the transformer model multiple times to generate candidate moves
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

        # If transformer gives too few useful moves, mix in random legal ones
        legal_moves = list(board.legal_moves)
        if len(candidates) < 6:
            # Add random legal moves if the model produces too few valid moves
            extra = random.sample(legal_moves, k=min(10, len(legal_moves)))
            candidates.extend(extra)

        # Remove duplicates preserving order
        seen = set()
        uniq = []
        for m in candidates:
            if m not in seen:
                seen.add(m)
                uniq.append(m)

        return uniq

    def _score_move_2ply(self, board: chess.Board, move: chess.Move, pov: bool) -> int:
    # Avoid moves an opponent can immediately punish
        b1 = board.copy()
        b1.push(move)

        if b1.is_checkmate():
            return 10**9

        if b1.is_stalemate() or b1.is_insufficient_material():
            return 0

        base = move_heuristic(board, move, pov)
        current_eval = evaluate_board(b1, pov)

        opp_moves = list(b1.legal_moves)
        if not opp_moves:
            return base + current_eval

        # Limit opponent replies for speed: prioritize tactical/legal replies
        scored_replies = []
        for reply in opp_moves:
            reply_score = 0

            if b1.is_capture(reply):
                captured = b1.piece_at(reply.to_square)
                mover = b1.piece_at(reply.from_square)
                reply_score += piece_value(captured.piece_type) if captured else 100
                reply_score -= piece_value(mover.piece_type) // 10 if mover else 0

            if b1.gives_check(reply):
                reply_score += 40
            if reply.promotion:
                reply_score += 500

            scored_replies.append((reply_score, reply))

        scored_replies.sort(key=lambda x: x[0], reverse=True)
        top_replies = [mv for _, mv in scored_replies[:8]]

        worst_for_us = 10**9
        for reply in top_replies:
            b2 = b1.copy()
            b2.push(reply)

            if b2.is_checkmate():
                # opponent mates us after our move
                return -10**9

            score_after_reply = evaluate_board(b2, pov)
            worst_for_us = min(worst_for_us, score_after_reply)

        return base + worst_for_us

    # Main function called by the tournament to choose a move
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        pov = board.turn

        if board.fullmove_number == 1:
            # Simple preferred opening moves
            preferred = ["e2e4", "d2d4", "g1f3", "c2c4"] if pov == chess.WHITE else ["e7e5", "d7d5", "g8f6", "c7c5"]
            for uci in preferred:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    return mv.uci()

        candidates = self._generate_candidates(board, fen)

        # Rank moves quickly using heuristics
        pre_ranked = []
        for mv in candidates:
            score = move_heuristic(board, mv, pov)
            pre_ranked.append((score, mv))

        pre_ranked.sort(key=lambda x: x[0], reverse=True)

        # Search only best few candidates more deeply
        top_candidates = [mv for _, mv in pre_ranked[:10]]

        best_move = None
        best_score = -10**18

        for mv in top_candidates:
            score = self._score_move_2ply(board, mv, pov)
            if score > best_score:
                best_score = score
                best_move = mv

        if best_move:
            return best_move.uci()


        return random.choice(legal_moves).uci()
