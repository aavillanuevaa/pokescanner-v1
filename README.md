# Pokémon Card Scanner (OpenCV + CLIP)

Real-time Pokemon card scanner built with **OpenCV** and **CLIP embeddings** 
Detects a card in a webcam feed, stabilizes the view, and identifies the card using visual similarity.
Currently only works with pokemon set: "Mega Evolutions"

This project is optimized for:
- Low-quality webcams
- Low latency
- Stable detection before recognition
- Manual lock/reset UI controls

# Features

- Real-time webcam scanning
- CLIP-based visual recognition
- Stability gating to prevent flicker
- ROI verification for better accuracy
- Live inset preview of detected card
- Optional UI mirroring

# Project Structure

├── app_ui.py          # UI loop, drawing, input handling
├── scanner_core.py    # Camera, detection, recognition logic
├── cv_card.py         # Card quad detection + warping
├── embedder.py        # CLIP embedding utilities
├── ref/
│   ├── README.md      # Instructions for reference data (not tracked)
│   ├── images/        # Card images (ignored by git)
│   ├── embeddings.npy # CLIP embeddings (ignored by git)
│   ├── cards.json     # Card metadata (ignored by git)
│   └── ids.json       # ID mapping (ignored by git)
├── requirements.txt
└── README.md

# Future Works

Planning to move on to work with all sets from SV era
Display market price 
