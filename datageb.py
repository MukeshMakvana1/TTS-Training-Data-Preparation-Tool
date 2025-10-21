import gradio as gr
import os
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import speech_recognition as sr
from pathlib import Path
import shutil
import json
import librosa
import numpy as np
import soundfile as sf
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

def process_audio(audio_file, language, chunk_duration, normalize_audio, remove_silence, 
                  apply_noise_reduction, min_duration, max_duration, progress=gr.Progress()):
    """Process audio file: split into chunks and transcribe each"""
    if audio_file is None:
        return "कृपया पहले एक ऑडियो फ़ाइल अपलोड करें! / Please upload an audio file first!", None, None, None
    
    try:
        output_dir = "tts_training_data"
        audio_chunks_dir = os.path.join(output_dir, "audio_chunks")
        filtered_chunks_dir = os.path.join(output_dir, "filtered_chunks")
        stats_dir = os.path.join(output_dir, "statistics")
        os.makedirs(audio_chunks_dir, exist_ok=True)
        os.makedirs(filtered_chunks_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        progress(0, desc="ऑडियो फ़ाइल लोड हो रही है... / Loading audio file...")
        
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(22050)
        
        if normalize_audio:
            progress(0.05, desc="ऑडियो नॉर्मलाइज़ हो रहा है... / Normalizing audio...")
            audio = normalize(audio)
            audio = compress_dynamic_range(audio)
        
        duration_ms = len(audio)
        chunk_length_ms = int(chunk_duration * 1000)
        num_chunks = (duration_ms // chunk_length_ms) + (1 if duration_ms % chunk_length_ms > 0 else 0)
        
        progress(0.1, desc=f"{num_chunks} चंक्स प्रोसेस हो रहे हैं... / Processing {num_chunks} chunks...")
        
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        lang_code = "hi-IN" if language == "Hindi (हिंदी)" else "en-US"
        
        transcriptions = []
        successful_chunks = 0
        failed_chunks = 0
        filtered_out = 0
        chunk_stats = []
        
        for i in range(num_chunks):
            start_ms = i * chunk_length_ms
            end_ms = min((i + 1) * chunk_length_ms, duration_ms)
            
            chunk = audio[start_ms:end_ms]
            chunk_duration_sec = len(chunk) / 1000.0
            
            if chunk_duration_sec < 1.0:
                continue
            
            chunk_filename = f"chunk_{i:04d}.wav"
            chunk_path = os.path.join(audio_chunks_dir, chunk_filename)
            chunk.export(chunk_path, format="wav")
            
            y, sr_rate = librosa.load(chunk_path, sr=22050)
            
            if apply_noise_reduction:
                y = apply_spectral_gating(y, sr_rate)
            
            if remove_silence:
                y_trimmed, _ = librosa.effects.trim(y, top_db=20)
                if len(y_trimmed) > 0:
                    y = y_trimmed
            
            rms_energy = float(np.sqrt(np.mean(y**2)))
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_rate)))
            
            sf.write(chunk_path, y, sr_rate)
            
            actual_duration = len(y) / sr_rate
            if actual_duration < min_duration or actual_duration > max_duration:
                filtered_out += 1
                continue
            
            try:
                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language=lang_code)
                    
                    if text.strip():
                        word_count = len(text.split())
                        
                        filtered_chunk_path = os.path.join(filtered_chunks_dir, chunk_filename)
                        shutil.copy(chunk_path, filtered_chunk_path)
                        
                        transcriptions.append(f"{chunk_filename}|{text}\n")
                        successful_chunks += 1
                        
                        chunk_stats.append({
                            "filename": chunk_filename,
                            "duration": actual_duration,
                            "transcription": text,
                            "word_count": word_count,
                            "rms_energy": rms_energy,
                            "zero_crossing_rate": zero_crossing_rate,
                            "spectral_centroid": spectral_centroid
                        })
                    else:
                        failed_chunks += 1
                        
            except sr.UnknownValueError:
                transcriptions.append(f"{chunk_filename}|[UNINTELLIGIBLE]\n")
                failed_chunks += 1
            except sr.RequestError as e:
                transcriptions.append(f"{chunk_filename}|[ERROR: {str(e)}]\n")
                failed_chunks += 1
            except Exception as e:
                transcriptions.append(f"{chunk_filename}|[ERROR: {str(e)}]\n")
                failed_chunks += 1
            
            progress((i + 1) / num_chunks, desc=f"चंक {i + 1}/{num_chunks} प्रोसेस हो रहा है... / Processing chunk {i + 1}/{num_chunks}")
        
        transcript_file = os.path.join(output_dir, "transcriptions.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.writelines(transcriptions)
        
        metadata_file = os.path.join(output_dir, "metadata.csv")
        with open(metadata_file, "w", encoding="utf-8") as f:
            for line in transcriptions:
                if "|" in line and "[" not in line:
                    f.write(line)
        
        stats_file = os.path.join(stats_dir, "audio_statistics.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(chunk_stats, f, indent=2, ensure_ascii=False)
        
        analysis_file = generate_analysis_report(chunk_stats, stats_dir)
        
        summary = f"""
✅ प्रोसेसिंग पूरी हुई! / Processing Complete!

📊 सांख्यिकी / Statistics:
- कुल चंक्स बनाए गए / Total chunks created: {num_chunks}
- सफलतापूर्वक ट्रांसक्राइब किए गए / Successfully transcribed: {successful_chunks}
- असफल/अस्पष्ट / Failed/Unintelligible: {failed_chunks}
- फ़िल्टर किए गए (अवधि) / Filtered out (duration): {filtered_out}
- कुल ऑडियो अवधि / Total audio duration: {duration_ms / 1000:.2f} seconds
- औसत चंक अवधि / Average chunk duration: {np.mean([s['duration'] for s in chunk_stats]):.2f}s (if chunk_stats else 0)

📁 आउटपुट स्थान / Output Location:
- सभी ऑडियो चंक्स / All audio chunks: {audio_chunks_dir}
- फ़िल्टर किए गए चंक्स / Filtered chunks: {filtered_chunks_dir}
- ट्रांसक्रिप्शन / Transcriptions: {transcript_file}
- मेटाडेटा (ट्रेनिंग के लिए) / Metadata (for training): {metadata_file}

💡 फ़ाइलें TTS मॉडल ट्रेनिंग के लिए तैयार हैं! / Files are ready for TTS model training!
        """
        
        return summary, transcript_file, metadata_file, stats_file
        
    except Exception as e:
        return f"❌ त्रुटि / Error: {str(e)}", None, None, None


def apply_spectral_gating(audio, sr, n_fft=2048, hop_length=512):
    """Simple noise reduction using spectral gating"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
    mask = magnitude > (noise_floor * 1.5)
    stft_denoised = stft * mask
    audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)
    return audio_denoised


def generate_analysis_report(chunk_stats, stats_dir):
    """Generate detailed analysis report"""
    if not chunk_stats:
        return None
    
    report_file = os.path.join(stats_dir, "analysis_report.txt")
    
    durations = [s['duration'] for s in chunk_stats]
    word_counts = [s['word_count'] for s in chunk_stats]
    energies = [s['rms_energy'] for s in chunk_stats]
    
    report = f"""
🔍 विस्तृत विश्लेषण रिपोर्ट / Detailed Analysis Report
{'='*60}

📊 अवधि विश्लेषण / Duration Analysis:
- औसत / Average: {np.mean(durations):.2f}s
- न्यूनतम / Minimum: {np.min(durations):.2f}s
- अधिकतम / Maximum: {np.max(durations):.2f}s
- मानक विचलन / Std Dev: {np.std(durations):.2f}s

📝 शब्द गणना विश्लेषण / Word Count Analysis:
- औसत शब्द प्रति चंक / Average words per chunk: {np.mean(word_counts):.1f}
- न्यूनतम / Minimum: {np.min(word_counts)}
- अधिकतम / Maximum: {np.max(word_counts)}

🔊 ऑडियो गुणवत्ता / Audio Quality:
- औसत RMS ऊर्जा / Average RMS Energy: {np.mean(energies):.4f}
- ऊर्जा स्थिरता / Energy Consistency: {'उच्च / High' if np.std(energies) < 0.01 else 'मध्यम / Medium'}

💡 सिफारिशें / Recommendations:
{'- अच्छी गुणवत्ता, प्रशिक्षण के लिए तैयार / Good quality, ready for training' if np.mean(energies) > 0.01 else '- कम ऊर्जा का पता चला, वॉल्यूम बढ़ाने पर विचार करें / Low energy detected, consider increasing volume'}
"""
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report_file


# Function 1: Validate Dataset
def validate_dataset(progress=gr.Progress()):
    """Validate the dataset for TTS training"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! पहले ऑडियो प्रोसेस करें। / No dataset found! Process audio first."
    
    progress(0, desc="डेटासेट मान्य हो रहा है... / Validating dataset...")
    
    issues = []
    valid_count = 0
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            audio_path = os.path.join(output_dir, "filtered_chunks", filename)
            
            if not os.path.exists(audio_path):
                issues.append(f"❌ लाइन {i+1}: ऑडियो फ़ाइल नहीं मिली / Line {i+1}: Audio file not found - {filename}")
            elif not text.strip():
                issues.append(f"⚠️ लाइन {i+1}: खाली ट्रांसक्रिप्शन / Line {i+1}: Empty transcription - {filename}")
            elif len(text.split()) < 2:
                issues.append(f"⚠️ लाइन {i+1}: बहुत छोटा ट्रांसक्रिप्शन / Line {i+1}: Very short transcription - {filename}")
            else:
                valid_count += 1
        
        progress((i + 1) / len(lines))
    
    report = f"""
✅ डेटासेट मान्यता पूर्ण / Dataset Validation Complete!

📊 परिणाम / Results:
- कुल प्रविष्टियाँ / Total entries: {len(lines)}
- मान्य प्रविष्टियाँ / Valid entries: {valid_count}
- समस्याएँ मिलीं / Issues found: {len(issues)}

{'🎉 डेटासेट प्रशिक्षण के लिए तैयार है! / Dataset is ready for training!' if len(issues) == 0 else '⚠️ कुछ समस्याएँ मिलीं / Some issues found:'}

{chr(10).join(issues[:10])}
{'...(और अधिक / and more)' if len(issues) > 10 else ''}
    """
    
    return report


# Function 2: Balance Dataset
def balance_dataset(min_samples, progress=gr.Progress()):
    """Balance dataset by removing over-represented samples"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="डेटासेट संतुलित हो रहा है... / Balancing dataset...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    word_freq = {}
    entries = []
    
    for line in lines:
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            words = text.lower().split()
            entries.append((filename, text, words))
            
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    balanced_entries = []
    word_usage = {}
    
    for filename, text, words in entries:
        should_add = True
        for word in words:
            if word_usage.get(word, 0) >= min_samples:
                should_add = False
                break
        
        if should_add:
            balanced_entries.append(f"{filename}|{text}\n")
            for word in words:
                word_usage[word] = word_usage.get(word, 0) + 1
    
    balanced_file = os.path.join(output_dir, "metadata_balanced.csv")
    with open(balanced_file, "w", encoding="utf-8") as f:
        f.writelines(balanced_entries)
    
    progress(1.0)
    
    report = f"""
✅ डेटासेट संतुलन पूर्ण / Dataset Balancing Complete!

📊 परिणाम / Results:
- मूल प्रविष्टियाँ / Original entries: {len(entries)}
- संतुलित प्रविष्टियाँ / Balanced entries: {len(balanced_entries)}
- हटाई गई / Removed: {len(entries) - len(balanced_entries)}

📁 संतुलित डेटासेट सहेजा गया / Balanced dataset saved: {balanced_file}
    """
    
    return report


# Function 3: Train/Val Split
def export_for_training(train_split, progress=gr.Progress()):
    """Split dataset into train and validation sets"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!", None, None
    
    progress(0, desc="ट्रेनिंग के लिए एक्सपोर्ट हो रहा है... / Exporting for training...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if "|" in l and "[" not in l]
    
    np.random.shuffle(lines)
    
    split_idx = int(len(lines) * train_split / 100)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    
    progress(1.0)
    
    summary = f"""
✅ एक्सपोर्ट पूर्ण / Export Complete!

📊 विभाजन / Split:
- प्रशिक्षण सेट / Training set: {len(train_lines)} samples
- सत्यापन सेट / Validation set: {len(val_lines)} samples
- अनुपात / Ratio: {train_split}% / {100-train_split}%

📁 फ़ाइलें / Files:
- {train_file}
- {val_file}

🚀 TTS मॉडल ट्रेनिंग शुरू करने के लिए तैयार! / Ready to start TTS model training!
    """
    
    return summary, train_file, val_file


# Function 4: Duplicate Detection
def detect_duplicates(progress=gr.Progress()):
    """Detect duplicate transcriptions"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="डुप्लीकेट का पता लगाया जा रहा है... / Detecting duplicates...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    transcriptions = {}
    duplicates = []
    
    for line in lines:
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            if text in transcriptions:
                duplicates.append(f"'{text}' - Files: {transcriptions[text]}, {filename}")
            else:
                transcriptions[text] = filename
    
    progress(1.0)
    
    report = f"""
✅ डुप्लीकेट जांच पूर्ण / Duplicate Check Complete!

📊 परिणाम / Results:
- कुल प्रविष्टियाँ / Total entries: {len(lines)}
- यूनिक ट्रांसक्रिप्शन / Unique transcriptions: {len(transcriptions)}
- डुप्लीकेट मिले / Duplicates found: {len(duplicates)}

{chr(10).join(duplicates[:15]) if duplicates else '🎉 कोई डुप्लीकेट नहीं मिला! / No duplicates found!'}
{'...(और अधिक / and more)' if len(duplicates) > 15 else ''}
    """
    
    return report


# Function 5: Character Count Analysis
def character_count_analysis(progress=gr.Progress()):
    """Analyze character count distribution"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="वर्ण गणना विश्लेषण... / Character count analysis...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    char_counts = []
    for line in lines:
        if "|" in line:
            _, text = line.strip().split("|", 1)
            char_counts.append(len(text))
    
    progress(1.0)
    
    report = f"""
✅ वर्ण गणना विश्लेषण पूर्ण / Character Count Analysis Complete!

📊 सांख्यिकी / Statistics:
- औसत वर्ण / Average characters: {np.mean(char_counts):.1f}
- न्यूनतम / Minimum: {np.min(char_counts)}
- अधिकतम / Maximum: {np.max(char_counts)}
- मानक विचलन / Std Dev: {np.std(char_counts):.1f}
- मध्यिका / Median: {np.median(char_counts):.1f}

💡 सिफारिश / Recommendation:
{'✅ अच्छा वितरण / Good distribution' if 50 < np.mean(char_counts) < 200 else '⚠️ वितरण की जांच करें / Check distribution'}
    """
    
    return report


# Function 6: Word Frequency Analysis
def word_frequency_analysis(top_n, progress=gr.Progress()):
    """Analyze most common words"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="शब्द आवृत्ति विश्लेषण... / Word frequency analysis...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    all_words = []
    for line in lines:
        if "|" in line:
            _, text = line.strip().split("|", 1)
            all_words.extend(text.lower().split())
    
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(top_n)
    
    progress(1.0)
    
    report = f"""
✅ शब्द आवृत्ति विश्लेषण पूर्ण / Word Frequency Analysis Complete!

📊 शीर्ष {top_n} शब्द / Top {top_n} Words:
{chr(10).join([f"{i+1}. '{word}' - {count} बार / times" for i, (word, count) in enumerate(most_common)])}

💡 कुल यूनिक शब्द / Total unique words: {len(word_freq)}
    """
    
    return report


# Function 7: Remove Short Samples
def remove_short_samples(min_words, progress=gr.Progress()):
    """Remove samples with fewer than minimum words"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="छोटे नमूने हटाए जा रहे हैं... / Removing short samples...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    filtered_lines = []
    removed_count = 0
    
    for line in lines:
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            if len(text.split()) >= min_words:
                filtered_lines.append(line)
            else:
                removed_count += 1
    
    filtered_file = os.path.join(output_dir, "metadata_filtered.csv")
    with open(filtered_file, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)
    
    progress(1.0)
    
    report = f"""
✅ फ़िल्टरिंग पूर्ण / Filtering Complete!

📊 परिणाम / Results:
- मूल नमूने / Original samples: {len(lines)}
- फ़िल्टर किए गए नमूने / Filtered samples: {len(filtered_lines)}
- हटाए गए / Removed: {removed_count}

📁 फ़ाइल सहेजी गई / File saved: {filtered_file}
    """
    
    return report


# Function 8: Generate Dataset Report
def generate_dataset_report(progress=gr.Progress()):
    """Generate comprehensive dataset report"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="रिपोर्ट जनरेट हो रही है... / Generating report...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    total_words = 0
    total_chars = 0
    durations = []
    
    for line in lines:
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            total_words += len(text.split())
            total_chars += len(text)
    
    progress(0.5)
    
    report_file = os.path.join(output_dir, "dataset_report.txt")
    report_content = f"""
{'='*70}
📊 संपूर्ण डेटासेट रिपोर्ट / COMPREHENSIVE DATASET REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 BASIC STATISTICS / बुनियादी आँकड़े:
- Total Samples / कुल नमूने: {total_samples}
- Total Words / कुल शब्द: {total_words}
- Total Characters / कुल वर्ण: {total_chars}
- Avg Words per Sample / औसत शब्द प्रति नमूना: {total_words/total_samples if total_samples > 0 else 0:.1f}
- Avg Characters per Sample / औसत वर्ण प्रति नमूना: {total_chars/total_samples if total_samples > 0 else 0:.1f}

🎯 DATASET READINESS / डेटासेट तैयारी:
✅ Ready for TTS Training / TTS प्रशिक्षण के लिए तैयार

📁 OUTPUT FILES / आउटपुट फ़ाइलें:
- Metadata: {metadata_file}
- Audio Chunks: {os.path.join(output_dir, 'filtered_chunks')}

💡 RECOMMENDATIONS / सिफारिशें:
- Validate dataset before training / प्रशिक्षण से पहले डेटासेट मान्य करें
- Split into train/val sets / ट्रेन/वैल सेट में विभाजित करें
- Check for duplicates / डुप्लीकेट की जांच करें

{'='*70}
Report saved to: {report_file}
{'='*70}
    """
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    progress(1.0)
    
    return report_content


# Function 9: Merge Multiple Datasets
def merge_datasets(progress=gr.Progress()):
    """Merge multiple metadata files"""
    output_dir = "tts_training_data"
    
    metadata_files = [
        os.path.join(output_dir, "metadata.csv"),
        os.path.join(output_dir, "metadata_balanced.csv"),
        os.path.join(output_dir, "metadata_filtered.csv")
    ]
    
    existing_files = [f for f in metadata_files if os.path.exists(f)]
    
    if len(existing_files) < 2:
        return "❌ कम से कम 2 मेटाडेटा फ़ाइलें चाहिए / Need at least 2 metadata files!"
    
    progress(0, desc="डेटासेट मर्ज हो रहे हैं... / Merging datasets...")
    
    all_lines = []
    for file in existing_files:
        with open(file, "r", encoding="utf-8") as f:
            all_lines.extend(f.readlines())
    
    # Remove duplicates
    unique_lines = list(set(all_lines))
    
    merged_file = os.path.join(output_dir, "metadata_merged.csv")
    with open(merged_file, "w", encoding="utf-8") as f:
        f.writelines(unique_lines)
    
    progress(1.0)
    
    report = f"""
✅ मर्ज पूर्ण / Merge Complete!

📊 परिणाम / Results:
- मर्ज की गई फ़ाइलें / Merged files: {len(existing_files)}
- कुल लाइनें / Total lines: {len(all_lines)}
- यूनिक लाइनें / Unique lines: {len(unique_lines)}
- डुप्लीकेट हटाए गए / Duplicates removed: {len(all_lines) - len(unique_lines)}

📁 मर्ज की गई फ़ाइल / Merged file: {merged_file}
    """
    
    return report


# Function 10: Export to JSON Format
def export_to_json(progress=gr.Progress()):
    """Export metadata to JSON format"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="JSON में एक्सपोर्ट हो रहा है... / Exporting to JSON...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    json_data = []
    for line in lines:
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            json_data.append({
                "audio_file": filename,
                "transcription": text,
                "word_count": len(text.split()),
                "char_count": len(text)
            })
    
    json_file = os.path.join(output_dir, "metadata.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    progress(1.0)
    
    return f"""
✅ JSON एक्सपोर्ट पूर्ण / JSON Export Complete!

📊 परिणाम / Results:
- कुल प्रविष्टियाँ / Total entries: {len(json_data)}
- फ़ाइल का आकार / File size: {os.path.getsize(json_file) / 1024:.2f} KB

📁 JSON फ़ाइल / JSON file: {json_file}
    """


# Function 11: Sample Random Entries
def sample_random_entries(num_samples, progress=gr.Progress()):
    """Sample random entries from dataset"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="रैंडम सैंपल निकाले जा रहे हैं... / Sampling random entries...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if "|" in l]
    
    if len(lines) < num_samples:
        num_samples = len(lines)
    
    sampled = np.random.choice(lines, num_samples, replace=False)
    
    sample_file = os.path.join(output_dir, "sample_dataset.csv")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.writelines(sampled)
    
    progress(1.0)
    
    preview = "\n".join([f"{i+1}. {line.strip()}" for i, line in enumerate(sampled[:5])])
    
    return f"""
✅ सैंपलिंग पूर्ण / Sampling Complete!

📊 परिणाम / Results:
- सैंपल किए गए / Sampled: {num_samples} entries

📝 पहले 5 एंट्रीज़ / First 5 Entries:
{preview}

📁 सैंपल फ़ाइल / Sample file: {sample_file}
    """


# Function 12: Calculate Dataset Size
def calculate_dataset_size(progress=gr.Progress()):
    """Calculate total dataset size"""
    output_dir = "tts_training_data"
    
    if not os.path.exists(output_dir):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="साइज़ की गणना हो रही है... / Calculating size...")
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
            file_count += 1
    
    progress(1.0)
    
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)
    
    return f"""
✅ साइज़ गणना पूर्ण / Size Calculation Complete!

📊 परिणाम / Results:
- कुल फ़ाइलें / Total files: {file_count}
- कुल साइज़ / Total size: {size_mb:.2f} MB ({size_gb:.3f} GB)
- औसत फ़ाइल साइज़ / Avg file size: {size_mb/file_count if file_count > 0 else 0:.2f} MB

💾 डिस्क स्पेस / Disk Space: {'✅ अच्छा / Good' if size_gb < 5 else '⚠️ बड़ा डेटासेट / Large dataset'}
    """


# Function 13: Clean Invalid Entries
def clean_invalid_entries(progress=gr.Progress()):
    """Remove entries with invalid/corrupted audio"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="अमान्य एंट्रीज़ साफ़ हो रही हैं... / Cleaning invalid entries...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    valid_lines = []
    invalid_count = 0
    
    for i, line in enumerate(lines):
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            audio_path = os.path.join(output_dir, "filtered_chunks", filename)
            
            try:
                # Try to load audio file
                audio = AudioSegment.from_file(audio_path)
                if len(audio) > 500:  # At least 0.5 seconds
                    valid_lines.append(line)
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
        
        progress((i + 1) / len(lines))
    
    cleaned_file = os.path.join(output_dir, "metadata_cleaned.csv")
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)
    
    progress(1.0)
    
    return f"""
✅ सफाई पूर्ण / Cleaning Complete!

📊 परिणाम / Results:
- मूल एंट्रीज़ / Original entries: {len(lines)}
- मान्य एंट्रीज़ / Valid entries: {len(valid_lines)}
- हटाई गई / Removed: {invalid_count}

📁 साफ़ की गई फ़ाइल / Cleaned file: {cleaned_file}
    """


# Function 14: Create Backup
def create_backup(progress=gr.Progress()):
    """Create backup of entire dataset"""
    output_dir = "tts_training_data"
    
    if not os.path.exists(output_dir):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="बैकअप बनाया जा रहा है... / Creating backup...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"tts_training_data_backup_{timestamp}"
    
    shutil.copytree(output_dir, backup_dir)
    
    progress(1.0)
    
    return f"""
✅ बैकअप पूर्ण / Backup Complete!

📁 बैकअप स्थान / Backup location: {backup_dir}
⏰ टाइमस्टैम्प / Timestamp: {timestamp}

💡 नोट / Note: मूल फ़ोल्डर सुरक्षित है / Original folder is safe
    """


# Function 15: Normalize All Audio
def normalize_all_audio(progress=gr.Progress()):
    """Normalize volume of all audio files"""
    output_dir = "tts_training_data"
    chunks_dir = os.path.join(output_dir, "filtered_chunks")
    
    if not os.path.exists(chunks_dir):
        return "❌ कोई ऑडियो चंक्स नहीं मिले! / No audio chunks found!"
    
    progress(0, desc="सभी ऑडियो नॉर्मलाइज़ हो रहे हैं... / Normalizing all audio...")
    
    audio_files = [f for f in os.listdir(chunks_dir) if f.endswith('.wav')]
    normalized_count = 0
    
    for i, filename in enumerate(audio_files):
        file_path = os.path.join(chunks_dir, filename)
        try:
            audio = AudioSegment.from_file(file_path)
            normalized_audio = normalize(audio)
            normalized_audio.export(file_path, format="wav")
            normalized_count += 1
        except:
            pass
        
        progress((i + 1) / len(audio_files))
    
    return f"""
✅ नॉर्मलाइज़ेशन पूर्ण / Normalization Complete!

📊 परिणाम / Results:
- कुल फ़ाइलें / Total files: {len(audio_files)}
- नॉर्मलाइज़ की गईं / Normalized: {normalized_count}
- विफल / Failed: {len(audio_files) - normalized_count}

🔊 सभी ऑडियो फ़ाइलें अब समान वॉल्यूम पर हैं! / All audio files now have consistent volume!
    """


# Function 16: Generate Statistics Chart
def generate_statistics_chart(progress=gr.Progress()):
    """Generate visual statistics chart"""
    output_dir = "tts_training_data"
    stats_file = os.path.join(output_dir, "statistics", "audio_statistics.json")
    
    if not os.path.exists(stats_file):
        return "❌ कोई सांख्यिकी फ़ाइल नहीं मिली! / No statistics file found!"
    
    progress(0, desc="चार्ट बना रहे हैं... / Generating chart...")
    
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    durations = [s['duration'] for s in stats]
    word_counts = [s['word_count'] for s in stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(durations, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Duration Distribution / अवधि वितरण')
    ax1.set_xlabel('Seconds / सेकंड')
    ax1.set_ylabel('Count / गणना')
    
    ax2.hist(word_counts, bins=20, color='lightgreen', edgecolor='black')
    ax2.set_title('Word Count Distribution / शब्द गणना वितरण')
    ax2.set_xlabel('Words / शब्द')
    ax2.set_ylabel('Count / गणना')
    
    plt.tight_layout()
    
    chart_file = os.path.join(output_dir, "statistics", "statistics_chart.png")
    plt.savefig(chart_file)
    plt.close()
    
    progress(1.0)
    
    return f"""
✅ चार्ट बनाया गया / Chart Generated!

📊 चार्ट फ़ाइल / Chart file: {chart_file}

💡 चार्ट में दिखाया गया है / Chart shows:
- अवधि वितरण / Duration distribution
- शब्द गणना वितरण / Word count distribution
    """


# Function 17: Find Long Samples
def find_long_samples(max_duration, progress=gr.Progress()):
    """Find samples longer than specified duration"""
    output_dir = "tts_training_data"
    stats_file = os.path.join(output_dir, "statistics", "audio_statistics.json")
    
    if not os.path.exists(stats_file):
        return "❌ कोई सांख्यिकी फ़ाइल नहीं मिली! / No statistics file found!"
    
    progress(0, desc="लंबे नमूने खोजे जा रहे हैं... / Finding long samples...")
    
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    long_samples = [s for s in stats if s['duration'] > max_duration]
    
    progress(1.0)
    
    if long_samples:
        preview = "\n".join([f"- {s['filename']}: {s['duration']:.2f}s" for s in long_samples[:10]])
        return f"""
✅ खोज पूर्ण / Search Complete!

📊 परिणाम / Results:
- {max_duration}s से लंबे नमूने / Samples longer than {max_duration}s: {len(long_samples)}

📝 पहले 10 नमूने / First 10 Samples:
{preview}
{'...(और अधिक / and more)' if len(long_samples) > 10 else ''}
        """
    else:
        return f"✅ कोई नमूना {max_duration}s से लंबा नहीं है! / No samples longer than {max_duration}s!"


# Function 18: Text Length Filter
def text_length_filter(min_chars, max_chars, progress=gr.Progress()):
    """Filter samples by text length"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="टेक्स्ट लंबाई से फ़िल्टर हो रहा है... / Filtering by text length...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    filtered_lines = []
    for line in lines:
        if "|" in line:
            filename, text = line.strip().split("|", 1)
            text_len = len(text)
            if min_chars <= text_len <= max_chars:
                filtered_lines.append(line)
    
    filtered_file = os.path.join(output_dir, "metadata_length_filtered.csv")
    with open(filtered_file, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)
    
    progress(1.0)
    
    return f"""
✅ फ़िल्टरिंग पूर्ण / Filtering Complete!

📊 परिणाम / Results:
- मूल नमूने / Original samples: {len(lines)}
- फ़िल्टर किए गए नमूने / Filtered samples: {len(filtered_lines)}
- हटाए गए / Removed: {len(lines) - len(filtered_lines)}

📏 लंबाई सीमा / Length range: {min_chars}-{max_chars} characters

📁 फ़ाइल सहेजी गई / File saved: {filtered_file}
    """


# Function 19: Create Dataset Splits
def create_dataset_splits(train_pct, val_pct, test_pct, progress=gr.Progress()):
    """Create train/val/test splits"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!", None, None, None
    
    if train_pct + val_pct + test_pct != 100:
        return "❌ प्रतिशत का योग 100 होना चाहिए! / Percentages must sum to 100!", None, None, None
    
    progress(0, desc="विभाजन बना रहे हैं... / Creating splits...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if "|" in l]
    
    np.random.shuffle(lines)
    
    train_idx = int(len(lines) * train_pct / 100)
    val_idx = train_idx + int(len(lines) * val_pct / 100)
    
    train_lines = lines[:train_idx]
    val_lines = lines[train_idx:val_idx]
    test_lines = lines[val_idx:]
    
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")
    test_file = os.path.join(output_dir, "test.txt")
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    with open(test_file, "w", encoding="utf-8") as f:
        f.writelines(test_lines)
    
    progress(1.0)
    
    summary = f"""
✅ विभाजन पूर्ण / Splits Complete!

📊 परिणाम / Results:
- प्रशिक्षण / Training: {len(train_lines)} ({train_pct}%)
- सत्यापन / Validation: {len(val_lines)} ({val_pct}%)
- परीक्षण / Test: {len(test_lines)} ({test_pct}%)

📁 फ़ाइलें / Files:
- {train_file}
- {val_file}
- {test_file}
    """
    
    return summary, train_file, val_file, test_file


# Function 20: Export Dataset Summary
def export_dataset_summary(progress=gr.Progress()):
    """Export comprehensive dataset summary"""
    output_dir = "tts_training_data"
    metadata_file = os.path.join(output_dir, "metadata.csv")
    stats_file = os.path.join(output_dir, "statistics", "audio_statistics.json")
    
    if not os.path.exists(metadata_file):
        return "❌ कोई डेटासेट नहीं मिला! / No dataset found!"
    
    progress(0, desc="सारांश एक्सपोर्ट हो रहा है... / Exporting summary...")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if "|" in l]
    
    summary_data = {
        "total_samples": len(lines),
        "total_duration": 0,
        "total_words": 0,
        "total_characters": 0,
        "avg_duration": 0,
        "avg_words": 0,
        "avg_characters": 0,
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for line in lines:
        if "|" in line:
            _, text = line.strip().split("|", 1)
            summary_data["total_words"] += len(text.split())
            summary_data["total_characters"] += len(text)
    
    if os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
            summary_data["total_duration"] = sum(s['duration'] for s in stats)
    
    if summary_data["total_samples"] > 0:
        summary_data["avg_duration"] = summary_data["total_duration"] / summary_data["total_samples"]
        summary_data["avg_words"] = summary_data["total_words"] / summary_data["total_samples"]
        summary_data["avg_characters"] = summary_data["total_characters"] / summary_data["total_samples"]
    
    summary_file = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    
    progress(1.0)
    
    return f"""
✅ सारांश एक्सपोर्ट पूर्ण / Summary Export Complete!

📊 डेटासेट सारांश / Dataset Summary:
- कुल नमूने / Total samples: {summary_data['total_samples']}
- कुल अवधि / Total duration: {summary_data['total_duration']:.2f}s
- कुल शब्द / Total words: {summary_data['total_words']}
- औसत अवधि / Avg duration: {summary_data['avg_duration']:.2f}s
- औसत शब्द / Avg words: {summary_data['avg_words']:.1f}
- औसत वर्ण / Avg characters: {summary_data['avg_characters']:.1f}

📁 सारांश फ़ाइल / Summary file: {summary_file}
    """


def clear_output():
    """Clear output directory"""
    output_dir = "tts_training_data"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    return "✅ आउटपुट फ़ोल्डर साफ़ हो गया! / Output folder cleared!", None, None, None


# Create Gradio interface
with gr.Blocks(title="TTS प्रशिक्षण डेटा तैयारी उपकरण / TTS Training Data Preparation Tool", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🎙️ TTS प्रशिक्षण डेटा तैयारी उपकरण / TTS Training Data Preparation Tool
    
    किसी भी लंबाई की ऑडियो फ़ाइल अपलोड करें! / Upload an audio file of any length!
    
    **समर्थित प्रारूप / Supported formats:** MP3, WAV, FLAC, OGG, M4A, और अधिक / and more!
    """)
    
    # YouTube Channel Promotion
    gr.Markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
        <h2 style="color: white; margin: 0;">🎬 Mind Hack Secrets</h2>
        <p style="color: white; margin: 10px 0;">AI, TTS, और Technology Tutorials के लिए मेरे YouTube चैनल को सब्सक्राइब करें!</p>
        <p style="color: white; margin: 10px 0;">Subscribe to my YouTube channel for AI, TTS, and Technology Tutorials!</p>
        <a href="https://www.youtube.com/@MindHackSecrets-o4y?sub_confirmation=1" target="_blank">
            <button style="background-color: #FF0000; color: white; padding: 15px 30px; font-size: 18px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                🔔 Subscribe Now / अभी सब्सक्राइब करें
            </button>
        </a>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ मुख्य सेटिंग्स / Main Settings")
            audio_input = gr.Audio(
                label="ऑडियो फ़ाइल अपलोड करें / Upload Audio File",
                type="filepath",
                sources=["upload"]
            )
            
            language = gr.Radio(
                choices=["Hindi (हिंदी)", "English (अंग्रेज़ी)"],
                value="Hindi (हिंदी)",
                label="भाषा / Language"
            )
            
            chunk_duration = gr.Slider(
                minimum=5,
                maximum=30,
                value=10,
                step=1,
                label="चंक अवधि (सेकंड) / Chunk Duration (seconds)"
            )
            
            gr.Markdown("### 🔧 उन्नत विकल्प / Advanced Options")
            
            normalize_audio = gr.Checkbox(
                label="ऑडियो नॉर्मलाइज़ करें / Normalize Audio",
                value=True
            )
            
            remove_silence = gr.Checkbox(
                label="मौन हटाएँ / Remove Silence",
                value=True
            )
            
            apply_noise_reduction = gr.Checkbox(
                label="शोर में कमी / Noise Reduction",
                value=False
            )
            
            with gr.Row():
                min_duration = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=0.5,
                    label="न्यूनतम अवधि / Min Duration (s)"
                )
                
                max_duration = gr.Slider(
                    minimum=10,
                    maximum=60,
                    value=30,
                    step=1,
                    label="अधिकतम अवधि / Max Duration (s)"
                )
            
            with gr.Row():
                process_btn = gr.Button("🚀 उत्पन्न करें / Generate & Process", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ साफ़ करें / Clear Output", variant="secondary")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="प्रोसेसिंग स्थिति / Processing Status",
                lines=20,
                max_lines=25
            )
            
            with gr.Row():
                transcript_file = gr.File(label="📄 ट्रांसक्रिप्शन / Transcriptions")
                metadata_file = gr.File(label="📋 मेटाडेटा / Metadata")
            
            stats_file = gr.File(label="📊 सांख्यिकी / Statistics")
    
    gr.Markdown("---")
    gr.Markdown("## 🛠️ 20 शक्तिशाली AI प्रशिक्षण कार्य / 20 Powerful AI Training Functions")
    
    with gr.Tabs():
        with gr.Tab("1️⃣ डेटासेट सत्यापन / Dataset Validation"):
            validate_btn = gr.Button("✅ सत्यापित करें / Validate", variant="primary")
            validation_output = gr.Textbox(label="परिणाम / Results", lines=12)
        
        with gr.Tab("2️⃣ डेटासेट संतुलन / Balance Dataset"):
            min_samples = gr.Slider(1, 20, 5, step=1, label="न्यूनतम नमूने / Min samples")
            balance_btn = gr.Button("⚖️ संतुलित करें / Balance", variant="primary")
            balance_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("3️⃣ ट्रेन/वैल विभाजन / Train/Val Split"):
            train_split = gr.Slider(60, 95, 80, step=5, label="प्रशिक्षण % / Training %")
            export_btn = gr.Button("📤 एक्सपोर्ट / Export", variant="primary")
            export_output = gr.Textbox(label="परिणाम / Results", lines=10)
            with gr.Row():
                train_file_out = gr.File(label="🎓 ट्रेन / Train")
                val_file_out = gr.File(label="✅ वैल / Val")
        
        with gr.Tab("4️⃣ डुप्लीकेट का पता लगाएं / Detect Duplicates"):
            detect_dup_btn = gr.Button("🔍 डुप्लीकेट खोजें / Find Duplicates", variant="primary")
            dup_output = gr.Textbox(label="परिणाम / Results", lines=12)
        
        with gr.Tab("5️⃣ वर्ण गणना विश्लेषण / Character Count"):
            char_btn = gr.Button("📊 विश्लेषण करें / Analyze", variant="primary")
            char_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("6️⃣ शब्द आवृत्ति / Word Frequency"):
            top_n = gr.Slider(5, 50, 20, step=5, label="शीर्ष शब्द / Top words")
            word_freq_btn = gr.Button("📈 विश्लेषण करें / Analyze", variant="primary")
            word_freq_output = gr.Textbox(label="परिणाम / Results", lines=12)
        
        with gr.Tab("7️⃣ छोटे नमूने हटाएं / Remove Short Samples"):
            min_words = gr.Slider(3, 20, 5, step=1, label="न्यूनतम शब्द / Min words")
            remove_short_btn = gr.Button("🗑️ हटाएं / Remove", variant="primary")
            remove_short_output = gr.Textbox(label="परिणाम / Results", lines=8)
        
        with gr.Tab("8️⃣ डेटासेट रिपोर्ट / Dataset Report"):
            report_btn = gr.Button("📋 रिपोर्ट बनाएं / Generate Report", variant="primary")
            report_output = gr.Textbox(label="परिणाम / Results", lines=15)
        
        with gr.Tab("9️⃣ डेटासेट मर्ज करें / Merge Datasets"):
            merge_btn = gr.Button("🔗 मर्ज करें / Merge", variant="primary")
            merge_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("🔟 JSON में एक्सपोर्ट / Export to JSON"):
            json_btn = gr.Button("📦 JSON एक्सपोर्ट / Export JSON", variant="primary")
            json_output = gr.Textbox(label="परिणाम / Results", lines=8)
        
        with gr.Tab("1️⃣1️⃣ रैंडम सैंपल / Random Sample"):
            num_samples = gr.Slider(5, 100, 20, step=5, label="नमूने / Samples")
            sample_btn = gr.Button("🎲 सैंपल करें / Sample", variant="primary")
            sample_output = gr.Textbox(label="परिणाम / Results", lines=12)
        
        with gr.Tab("1️⃣2️⃣ डेटासेट साइज़ / Dataset Size"):
            size_btn = gr.Button("💾 साइज़ गणना / Calculate Size", variant="primary")
            size_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("1️⃣3️⃣ अमान्य साफ़ करें / Clean Invalid"):
            clean_btn = gr.Button("🧹 साफ़ करें / Clean", variant="primary")
            clean_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("1️⃣4️⃣ बैकअप बनाएं / Create Backup"):
            backup_btn = gr.Button("💾 बैकअप / Backup", variant="primary")
            backup_output = gr.Textbox(label="परिणाम / Results", lines=8)
        
        with gr.Tab("1️⃣5️⃣ सभी नॉर्मलाइज़ करें / Normalize All"):
            normalize_btn = gr.Button("🔊 नॉर्मलाइज़ / Normalize", variant="primary")
            normalize_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("1️⃣6️⃣ चार्ट बनाएं / Generate Chart"):
            chart_btn = gr.Button("📊 चार्ट / Chart", variant="primary")
            chart_output = gr.Textbox(label="परिणाम / Results", lines=8)
        
        with gr.Tab("1️⃣7️⃣ लंबे नमूने खोजें / Find Long Samples"):
            max_dur = gr.Slider(10, 60, 20, step=5, label="अधिकतम अवधि / Max duration (s)")
            long_btn = gr.Button("🔎 खोजें / Find", variant="primary")
            long_output = gr.Textbox(label="परिणाम / Results", lines=12)
        
        with gr.Tab("1️⃣8️⃣ टेक्स्ट लंबाई फ़िल्टर / Text Length Filter"):
            with gr.Row():
                min_chars = gr.Slider(10, 200, 50, step=10, label="न्यूनतम वर्ण / Min chars")
                max_chars = gr.Slider(200, 1000, 500, step=50, label="अधिकतम वर्ण / Max chars")
            text_filter_btn = gr.Button("🔍 फ़िल्टर / Filter", variant="primary")
            text_filter_output = gr.Textbox(label="परिणाम / Results", lines=10)
        
        with gr.Tab("1️⃣9️⃣ 3-Way स्प्लिट / 3-Way Split"):
            with gr.Row():
                train_pct = gr.Slider(50, 90, 70, step=5, label="ट्रेन % / Train %")
                val_pct = gr.Slider(5, 30, 15, step=5, label="वैल % / Val %")
                test_pct = gr.Slider(5, 30, 15, step=5, label="टेस्ट % / Test %")
            split3_btn = gr.Button("📊 विभाजन / Split", variant="primary")
            split3_output = gr.Textbox(label="परिणाम / Results", lines=10)
            with gr.Row():
                train3_file = gr.File(label="ट्रेन / Train")
                val3_file = gr.File(label="वैल / Val")
                test3_file = gr.File(label="टेस्ट / Test")
        
        with gr.Tab("2️⃣0️⃣ सारांश एक्सपोर्ट / Export Summary"):
            summary_btn = gr.Button("📄 सारांश / Summary", variant="primary")
            summary_output = gr.Textbox(label="परिणाम / Results", lines=15)
    
    # Event handlers
    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, language, chunk_duration, normalize_audio, remove_silence, 
                apply_noise_reduction, min_duration, max_duration],
        outputs=[output_text, transcript_file, metadata_file, stats_file]
    )
    
    clear_btn.click(
        fn=clear_output,
        inputs=[],
        outputs=[output_text, transcript_file, metadata_file, stats_file]
    )
    
    validate_btn.click(fn=validate_dataset, outputs=[validation_output])
    balance_btn.click(fn=balance_dataset, inputs=[min_samples], outputs=[balance_output])
    export_btn.click(fn=export_for_training, inputs=[train_split], outputs=[export_output, train_file_out, val_file_out])
    detect_dup_btn.click(fn=detect_duplicates, outputs=[dup_output])
    char_btn.click(fn=character_count_analysis, outputs=[char_output])
    word_freq_btn.click(fn=word_frequency_analysis, inputs=[top_n], outputs=[word_freq_output])
    remove_short_btn.click(fn=remove_short_samples, inputs=[min_words], outputs=[remove_short_output])
    report_btn.click(fn=generate_dataset_report, outputs=[report_output])
    merge_btn.click(fn=merge_datasets, outputs=[merge_output])
    json_btn.click(fn=export_to_json, outputs=[json_output])
    sample_btn.click(fn=sample_random_entries, inputs=[num_samples], outputs=[sample_output])
    size_btn.click(fn=calculate_dataset_size, outputs=[size_output])
    clean_btn.click(fn=clean_invalid_entries, outputs=[clean_output])
    backup_btn.click(fn=create_backup, outputs=[backup_output])
    normalize_btn.click(fn=normalize_all_audio, outputs=[normalize_output])
    chart_btn.click(fn=generate_statistics_chart, outputs=[chart_output])
    long_btn.click(fn=find_long_samples, inputs=[max_dur], outputs=[long_output])
    text_filter_btn.click(fn=text_length_filter, inputs=[min_chars, max_chars], outputs=[text_filter_output])
    split3_btn.click(fn=create_dataset_splits, inputs=[train_pct, val_pct, test_pct], 
                     outputs=[split3_output, train3_file, val3_file, test3_file])
    summary_btn.click(fn=export_dataset_summary, outputs=[summary_output])
    
    gr.Markdown("""
    ---
    ### 📝 आउटपुट फ़ाइलों की व्याख्या / Output Files Explained:
    
    1. **audio_chunks/** - सभी चंक्स / All chunks
    2. **filtered_chunks/** - गुणवत्ता फ़िल्टर / Quality filtered
    3. **transcriptions.txt** - ट्रांसक्रिप्शन / Transcriptions
    4. **metadata.csv** - TTS ट्रेनिंग के लिए / For TTS training
    5. **statistics/** - विस्तृत आँकड़े / Detailed stats
    
    ### 💡 TTS मॉडल प्रशिक्षण वर्कफ्लो / TTS Model Training Workflow:
    1. ऑडियो प्रोसेस करें / Process Audio ✅
    2. डेटासेट सत्यापित करें / Validate Dataset ✅
    3. डुप्लीकेट हटाएं / Remove Duplicates 🔍
    4. डेटासेट संतुलित करें / Balance Dataset ⚖️
    5. ट्रेन/वैल में विभाजित करें / Split Train/Val 📤
    6. ट्रेनिंग शुरू करें! / Start Training! 🚀
    
    ### 🎬 YouTube चैनल / YouTube Channel:
    **Mind Hack Secrets** - AI और TTS Tutorials के लिए सब्सक्राइब करें!
    [https://www.youtube.com/@MindHackSecrets-o4y](https://www.youtube.com/@MindHackSecrets-o4y?sub_confirmation=1)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)