from dotenv import load_dotenv
load_dotenv()  # Load .env file

import os
print("HUGGINGFACEHUB_API_TOKEN =", os.getenv("HUGGINGFACEHUB_API_TOKEN"))  # Debug line
