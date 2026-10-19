"""Download SKAB CSVs without requiring Git or extracting an archive."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent

def download():
    request = Request('https://api.github.com/repos/waico/SKAB/git/trees/master?recursive=1',
                      headers={'User-Agent': 'sensor-anomaly-detector'})
    with urlopen(request, timeout=60) as response:
        tree = json.load(response)['tree']
    files = [entry['path'] for entry in tree
             if entry['path'].startswith('data/') and entry['path'].endswith('.csv')]
    if not files:
        raise RuntimeError('No CSV files found in the upstream repository.')
    for name in files:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        url = 'https://raw.githubusercontent.com/waico/SKAB/master/' + name
        with urlopen(url, timeout=60) as response:
            content = response.read()
        temporary = target.with_suffix('.part')
        temporary.write_bytes(content)
        temporary.replace(target)
        print('Downloaded', name)
    print(f'Ready: {len(files)} CSV files in {ROOT / "data"}')

if __name__ == '__main__':
    download()
