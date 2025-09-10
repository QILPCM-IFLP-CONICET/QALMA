import subprocess
import os

def build_docs():
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(docs_dir, '_build', 'html')
    # Ensure the build directory exists
    os.makedirs(build_dir, exist_ok=True)
    # Run sphinx-build
    subprocess.run([
        'sphinx-build',
        '-b', 'html',
        docs_dir,
        build_dir
    ], check=True)
    print(f"Documentation built in {build_dir}")

if __name__ == "__main__":
    build_docs()
