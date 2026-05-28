import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="Example: your-username/finnews-distilbert-sentiment")
    parser.add_argument("--model_dir", default="Fine-tuned_Model_files/finnews_distilbert_sentiment_upload")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN or pass --hf_token.")

    api = HfApi(token=args.hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned financial sentiment model",
    )

    print(f"Uploaded model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
