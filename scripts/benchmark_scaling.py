import json

from scripts.benchmark_efficiency import benchmark


def main():
    results = []
    for fraction in (.2, .4, .6, .8, 1.0):
        result = benchmark()
        result['catalog_fraction'] = fraction
        result['note'] = 'Demo smoke output; use explicit Gowalla subset manifests for paper scaling.'
        results.append(result)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
