import { Chessground } from "@lichess-org/chessground";
import { Chess } from "chess.js";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

const chess = new Chess();

function computeDests(chess) {
  const dests = new Map();
  
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];
  
  files.forEach(file => {
    ranks.forEach(rank => {
      const square = file + rank;
      const moves = chess.moves({ square, verbose: true });
      if (moves.length) {
        dests.set(square, moves.map(m => m.to));
      }
    });
  });

  return dests;
}

function toColor(chess) {
  return chess.turn() === 'w' ? 'white' : 'black';
}

const ground = Chessground(document.getElementById("board"), {
  orientation: "white",
  movable: {
    free: false,
    color: "white",
    dests: computeDests(chess),
    events: {
      after: async (orig, dest) => {
        // Make the user's move
        const move = chess.move({
          from: orig,
          to: dest,
          promotion: "q"
        });
        
        if (!move) return;

        // Update board after user move
        ground.set({
          fen: chess.fen(),
          turnColor: toColor(chess),
          movable: {
            color: toColor(chess),
            dests: computeDests(chess)
          }
        });

        // Get move history
        const history = chess.history({ verbose: true })
          .map(m => m.from + m.to);

        console.log("Sending moves to AI:", history);

        try {
          // Request AI move
          const res = await fetch("http://localhost:8000/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ moves: history })
          });

          const data = await res.json();
          console.log("AI response:", data);

          if (data.move) {
            // Make AI move
            const aiMove = chess.move({
              from: data.move.slice(0, 2),
              to: data.move.slice(2, 4),
              promotion: data.move.length > 4 ? data.move[4] : undefined
            });

            if (aiMove) {
              console.log("AI played:", data.move);
              
              // Update board after AI move
              ground.set({
                fen: chess.fen(),
                turnColor: toColor(chess),
                movable: {
                  color: toColor(chess),
                  dests: computeDests(chess)
                },
                lastMove: [data.move.slice(0, 2), data.move.slice(2, 4)]
              });

              // Check for game over
              if (chess.isGameOver()) {
                let result = "Game Over! ";
                if (chess.isCheckmate()) {
                  result += chess.turn() === 'w' ? "Black wins by checkmate!" : "White wins by checkmate!";
                } else if (chess.isDraw()) {
                  result += "Draw!";
                } else if (chess.isStalemate()) {
                  result += "Stalemate!";
                }
                alert(result);
              }
            }
          } else {
            console.error("No move returned from AI");
            alert("AI couldn't find a move!");
          }
        } catch (error) {
          console.error("Error communicating with AI:", error);
          alert("Error: Could not reach the chess AI server. Make sure it's running on http://localhost:8000");
        }
      }
    }
  }
});