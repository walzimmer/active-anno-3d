import re
import os

def parse_log_file(filepath, get_evals=False):
    
    # Regular expression pattern to match a timestamp followed by "INFO"
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}   INFO"

    entries = []

    with open(filepath, 'r') as f:
        current_entry = None
        for line in f:
            if re.match(pattern, line):
                # If we encounter a new pattern and current_entry is not None, add it to the list
                if current_entry is not None:
                    entries.append(current_entry.strip())
                # Reset current entry with data after the pattern
                current_entry = line.split('INFO')[-1].strip()
            elif current_entry is not None:
                # Append subsequent lines to current_entry
                current_entry += " " + line.strip()

    # Adding the last entry if it exists
    if current_entry:
        entries.append(current_entry.strip())
    
    if get_evals:
        pattern = r'Generate label finished\(sec_per_example: \d+\.\d+ second\)\.'
        matching_indices = [i for i, item in enumerate(entries) if re.match(pattern, item)]

        eval_nums = entries[matching_indices[0]+8:-2]
        eval_dict = parse_evaluation(txt=eval_nums[0])
        
    return entries, eval_dict

def parse_evaluation(txt):
        lines = txt.split("|")

        # Removing empty lines and extra whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Extracting categories and their mAP values
        categories = []
        overall = []
        close_range = []
        mid_range = []
        far_range = []
        occurences = []
        
        for i in range(6, len(lines), 6):  # Starting from 1 to skip the header
            category = re.search(r'\w+', lines[i]).group()
            categories.append(category)
            overall.append(float(lines[i+1].strip()))
            close_range.append(float(lines[i+2].split()[0]))
            mid_range.append(float(lines[i+3].split()[0]))
            far_range.append(float(lines[i+4].split()[0]))
            occurences.append(lines[i+5].split()[0])
        
        eval_dict = {
            'Categories': categories,
            'Overall': overall,
            '0-30m': close_range,
            '30-50m': mid_range,
            '50m-inf': far_range,
            'Occurences': occurences
        }

        return eval_dict