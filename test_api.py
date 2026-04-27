import urllib.request, urllib.parse, json, os

# Load key from .env
def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env

env = load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
API_KEY = env.get("FLIGHT_API_KEY", "")
print(f"Key loaded: [{API_KEY}]  length={len(API_KEY)}")

url = 'http://api.aviationstack.com/v1/flights?' + urllib.parse.urlencode({
    'access_key': API_KEY,
    'dep_iata': 'LHR',
    'flight_status': 'scheduled',
    'limit': 3
})

try:
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.loads(r.read())
    if 'error' in data:
        print('API ERROR:', data['error'])
    else:
        flights = data.get('data', [])
        print(f'API OK — got {len(flights)} flights')
        for f in flights[:3]:
            dep = f.get('departure', {})
            arr = f.get('arrival', {})
            fn  = f.get('flight', {}).get('iata', '?')
            print(f"  {fn} | {dep.get('iata')} T:{dep.get('terminal')} -> {arr.get('iata')} T:{arr.get('terminal')} | delay={dep.get('delay')}")
except Exception as e:
    print('ERROR:', e)
