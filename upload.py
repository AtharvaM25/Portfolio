from huggingface_hub import HfApi

api = HfApi()

# Replace with your details
repo_id = "YOUR_USERNAME/YOUR_SPACE_NAME"
token = "YOUR_HF_WRITE_TOKEN"

print("Uploading...")
api.upload_folder(
    folder_path=".",              # Uploads current folder
    repo_id=repo_id,
    repo_type="space",
    token=token
)
print("Done!")