import re
import random
import chess
import torch

from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament import Player

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def material_score(board: chess.Board, pov: bool) -> int:
    """Material from pov's perspective."""
    score = 0
    for ptype, val in PIECE_VALUE.items():
        score += val * len(board.pieces(ptype, pov))
        score -= val * len(board.pieces(ptype, not pov))
    return score

class TransformerPlayer(Player):
    def __init__(self, name: str, hf_model_id: str = "Chaunce1121/chess-fen-move-model", num_tries: int = 8):
        super().__init__(name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.num_tries = num_tries

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMOVE:"

    def _extract(self, txt: str) -> List[str]:
        tail = txt.split("MOVE:", 1)[-1].lower()
        # matches e2e4 or e7e8q etc.
        return re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", tail)

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        # Simple opening rule (first move)
        if board.fullmove_number == 1:
            if board.turn == chess.WHITE:
                for move in ["e2e4", "d2d4"]:
                    mv = chess.Move.from_uci(move)
                    if mv in board.legal_moves:
                        return mv.uci()
            else:
                for move in ["e7e5", "d7d5"]:
                    mv = chess.Move.from_uci(move)
                    if mv in board.legal_moves:
                        return mv.uci()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        pov = board.turn  # side we are playing
        prompt = self._prompt(fen)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # collect candidates from the model
        candidates: List[chess.Move] = []
        for _ in range(self.num_tries):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=12,
                    do_sample=True,
                    top_k=80,
                    temperature=0.9,
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

        # If model gave nothing, add a few random legal options
        if not candidates:
            candidates = random.sample(legal_moves, k=min(10, len(legal_moves)))

        # Prefer immediate checkmate if available
        best_move = None
        best_score = -10**9

        # Remove duplicates while preserving order
        seen = set()
        uniq = []
        for m in candidates:
            if m not in seen:
                seen.add(m)
                uniq.append(m)

        for mv in uniq:
            b2 = board.copy()
            b2.push(mv)

            if b2.is_checkmate():
                return mv.uci()

            # simple “tactical” bias: captures & checks get a small bonus
            score = material_score(b2, pov)
            if board.is_capture(mv):
                score += 1
            if board.gives_check(mv):
                score += 1
           # Penalize moves where the piece can immediately be captured
            if b2.is_attacked_by(not pov, mv.to_square):
                score -= 2

            if score > best_score:
                best_score = score
                best_move = mv

        return best_move.uci() if best_move else random.choice(legal_moves).uci()