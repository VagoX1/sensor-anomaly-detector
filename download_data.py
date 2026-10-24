"""Download SKAB CSVs without requiring Git or extracting an archive."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

# Resolve paths relative to this script, regardless of the terminal's current folder.
ROOT = Path(__file__).resolve().parent

def download():
    # GitHub's recursive tree lists repository paths without downloading the whole repo.
    request = Request('https://api.github.com/repos/waico/SKAB/git/trees/master?recursive=1',
                      headers={'User-Agent': 'sensor-anomaly-detector'})
    with urlopen(request, timeout=60) as response:
        tree = json.load(response)['tree']
    # Keep only dataset CSVs; ignore code, images, and other repository files.
    files = [entry['path'] for entry in tree
             if entry['path'].startswith('data/') and entry['path'].endswith('.csv')]
    if not files:
        raise RuntimeError('No CSV files found in the upstream repository.')
    for name in files:
        # Preserve upstream subfolders so experiments with identical filenames stay separate.
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Re-running the downloader reuses files already present locally.
            continue
        url = 'https://raw.githubusercontent.com/waico/SKAB/master/' + name
        with urlopen(url, timeout=60) as response:
            content = response.read()
        # Write to a temporary filename first, then rename the completed file.
        # An interrupted write therefore does not leave a partial CSV at the final path.
        temporary = target.with_suffix('.part')
        temporary.write_bytes(content)
        temporary.replace(target)
        print('Downloaded', name)
    print(f'Ready: {len(files)} CSV files in {ROOT / "data"}')

if __name__ == '__main__':
    # Download only when executed directly, not when imported by another module.
    download()
