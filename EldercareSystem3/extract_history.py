import os
import json

brain_dir = r"C:\Users\arthu\.gemini\antigravity\brain"
target_files = ["train_custom_ai.py", "skeleton_action_model.py"]

versions = {tf: [] for tf in target_files}

for conv_dir in os.listdir(brain_dir):
    log_path = os.path.join(brain_dir, conv_dir, ".system_generated", "logs", "transcript.jsonl")
    if not os.path.exists(log_path):
        continue
    
    print(f"Scanning {conv_dir}...")
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                if 'tool_calls' in d and d['tool_calls']:
                    for tc in d['tool_calls']:
                        if 'write_to_file' in tc.get('name', ''):
                            args = tc.get('args', {})
                            target = args.get('TargetFile', '')
                            content = args.get('CodeContent', '')
                            try:
                                content = json.loads(content)
                            except:
                                pass
                            
                            for tf in target_files:
                                if tf in target:
                                    # Prevent exact duplicates
                                    if len(versions[tf]) == 0 or versions[tf][-1] != content:
                                        versions[tf].append(content)
            except Exception as e:
                pass

for tf, contents in versions.items():
    print(f"Found {len(contents)} versions for {tf}")
    for i, content in enumerate(contents):
        save_path = f"old_{tf.replace('.py', '')}_{i+1}.py"
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)

