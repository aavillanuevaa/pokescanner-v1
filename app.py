# app.py
from embedder import ClipEmbedder
from scanner_core import open_camera, load_reference, process_frame
from app_ui import run_app_loop
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"   # or "SILENT" if supported

def main():
    cards, ref_embs, ref_ids = load_reference()
    embedder = ClipEmbedder()
    cap = open_camera()
    run_app_loop(cap, embedder, cards, ref_embs, ref_ids, process_frame)


if __name__ == "__main__":
    main()
