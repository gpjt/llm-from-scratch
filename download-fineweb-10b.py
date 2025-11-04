import os
import time

from huggingface_hub import snapshot_download


def download_dataset(name):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    start = time.time()
    folder = snapshot_download(
        f"HuggingFaceFW/{name}",
        repo_type="dataset",
        local_dir=f"./{name}/",
        allow_patterns="sample/10BT/*"
    )
    end = time.time()
    print(f"\n\n\n\nDownloaded {name} to {folder} in {end - start} seconds")


def main():
    download_dataset("fineweb-edu")
    download_dataset("fineweb")


if __name__ == "__main__":
    main()
