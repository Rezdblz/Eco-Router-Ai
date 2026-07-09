import json

d = json.load(open('output/analytics.json'))
print("\nTask Routing Results:")
print("-" * 90)
print(f"{'Task':<6} {'Category':<32} {'Method':<12} {'Model':<20} {'Reason':<25}")
print("-" * 90)
for t in d['tasks']:
    model_name = t['chosen_model'].split('/')[-1] if '/' in t['chosen_model'] else t['chosen_model']
    print(f"{t['task_id']:<6} {t['category']:<32} {t['method']:<12} {model_name:<20} {t['reason']:<25}")

print("-" * 90)
print(f"\nMetrics:")
print(f"  Total tokens: {d['summary']['total_tokens']}")
print(f"  Router tokens: {d['summary']['router_total_tokens']}")
print(f"  Pipeline time: {d['summary']['pipeline_elapsed_seconds']}s")
print(f"  Average model time: {d['summary']['average_model_elapsed_seconds']}s")
