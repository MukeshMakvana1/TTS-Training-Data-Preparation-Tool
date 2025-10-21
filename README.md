# TTS-Training-Data-Preparation-Tool
TTS Training Data Preparation Tool** is a comprehensive Gradio-based application designed to process audio files and prepare high-quality datasets for Text-to-Speech (TTS) model training. It supports both **Hindi (हिंदी)** and **English** languages with a fully bilingual interface.
<img width="1222" height="680" alt="1" src="https://github.com/user-attachments/assets/2fda5b93-28e5-4f78-9777-ff1b6e64a9db" />
<img width="1193" height="659" alt="2" src="https://github.com/user-attachments/assets/6a629a4c-c677-4fb3-858e-baefdb94c6ad" />
<img width="1014" height="462" alt="3" src="https://github.com/user-attachments/assets/31ab0ad3-5ae8-4432-bf27-a4370f143aed" />

# 🎙️ TTS Training Data Preparation Tool - Complete Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Main Functions](#main-functions)
6. [20 AI Training Functions](#20-ai-training-functions)
7. [Output Files](#output-files)
8. [Workflow Guide](#workflow-guide)
9. [Troubleshooting](#troubleshooting)
10. [YouTube Channel](#youtube-channel)

---

## 🌟 Overview

**TTS Training Data Preparation Tool** is a comprehensive Gradio-based application designed to process audio files and prepare high-quality datasets for Text-to-Speech (TTS) model training. It supports both **Hindi (हिंदी)** and **English** languages with a fully bilingual interface.

### Purpose
- Process audio files of any length
- Split audio into manageable chunks (5-30 seconds)
- Transcribe audio automatically using Google Speech Recognition
- Apply advanced audio processing (normalization, noise reduction, silence removal)
- Provide 20 powerful functions for dataset management and analysis
- Export ready-to-use training data for TTS models

---

## ✨ Features

### Core Features
- ✅ **Bilingual Support**: Hindi (हिंदी) and English interface
- ✅ **Any Audio Format**: MP3, WAV, FLAC, OGG, M4A, and more
- ✅ **Automatic Chunking**: Split audio into customizable chunks (5-30 seconds)
- ✅ **Speech-to-Text**: Automatic transcription using Google Speech Recognition
- ✅ **Audio Enhancement**: Normalization, noise reduction, silence removal
- ✅ **Progress Tracking**: Real-time progress bars for all operations
- ✅ **Quality Filtering**: Duration and quality-based filtering
- ✅ **Detailed Statistics**: Comprehensive audio analysis and reporting

### Advanced Features
- 🔧 **20 AI Training Functions**: Dataset validation, balancing, splitting, merging, and more
- 📊 **Visual Analytics**: Generate statistical charts and reports
- 💾 **Multiple Export Formats**: TXT, CSV, JSON
- 🎯 **Production-Ready**: Generates train/validation/test splits
- 🔍 **Quality Control**: Duplicate detection, invalid entry removal
- 💡 **Smart Filtering**: Word count, character count, duration-based filtering

---

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- Internet connection (for speech recognition)
- FFmpeg (for audio processing)

### Step 1: Install Python Packages

```bash
pip install gradio pydub SpeechRecognition librosa soundfile matplotlib numpy
```

### Step 2: Install FFmpeg

#### Windows:
1. Download FFmpeg from: https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH: `C:\ffmpeg\bin`

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

### Step 3: Verify Installation

```bash
python -c "import gradio, pydub, speech_recognition, librosa, soundfile"
ffmpeg -version
```

---

## 🚀 Quick Start

### 1. Run the Application

```bash
python datageb.py
```

### 2. Open in Browser
The application will automatically open in your default browser at:
```
http://localhost:7860
```

### 3. Basic Workflow
1. **Upload Audio File**: Click "Upload Audio File" and select your audio
2. **Select Language**: Choose Hindi (हिंदी) or English (अंग्रेज़ी)
3. **Adjust Settings**: Configure chunk duration and audio processing options
4. **Click Generate**: Click "🚀 Generate & Process" button
5. **Download Files**: Download transcriptions, metadata, and statistics

---

## 🎛️ Main Functions

### Audio Processing Settings

#### 1. Language Selection
- **Hindi (हिंदी)**: For Hindi audio transcription
- **English (अंग्रेज़ी)**: For English audio transcription

#### 2. Chunk Duration
- **Range**: 5-30 seconds
- **Default**: 10 seconds
- **Purpose**: Controls the length of each audio segment

#### 3. Audio Normalization
- **Purpose**: Balances volume across all chunks
- **Benefit**: Consistent audio levels for training
- **Recommended**: Enable (Default: ON)

#### 4. Remove Silence
- **Purpose**: Trims silence from start and end of chunks
- **Benefit**: Removes dead air, focuses on speech
- **Recommended**: Enable (Default: ON)

#### 5. Noise Reduction
- **Purpose**: Reduces background noise using spectral gating
- **Benefit**: Cleaner audio for training
- **Note**: May slightly affect audio quality
- **Recommended**: Enable for noisy audio (Default: OFF)

#### 6. Duration Filters
- **Min Duration**: 1-10 seconds (Default: 3s)
- **Max Duration**: 10-60 seconds (Default: 30s)
- **Purpose**: Filter out chunks that are too short or too long

---

## 🛠️ 20 AI Training Functions

### Category 1: Data Validation & Quality (Functions 1-5)

#### 1. 📊 Dataset Validation
**Purpose**: Checks dataset for issues before training

**Checks for**:
- Missing audio files
- Empty transcriptions
- Very short transcriptions (< 2 words)
- File integrity

**Output**: Detailed validation report with issues list

**When to Use**: Always run before training

---

#### 2. ⚖️ Balance Dataset
**Purpose**: Removes over-represented samples to prevent model bias

**How it Works**:
- Counts word frequency across all samples
- Limits maximum occurrences per word
- Creates balanced dataset file

**Parameters**:
- `min_samples`: Maximum samples per word (Default: 5)

**Output**: `metadata_balanced.csv`

**When to Use**: When certain words appear too frequently

---

#### 3. 📤 Train/Validation Split
**Purpose**: Splits dataset into training and validation sets

**Parameters**:
- `train_split`: Percentage for training (Default: 80%)

**Output**:
- `train.txt`: Training dataset
- `val.txt`: Validation dataset

**When to Use**: Before starting model training

---

#### 4. 🔍 Detect Duplicates
**Purpose**: Finds duplicate transcriptions in dataset

**How it Works**:
- Compares all transcriptions
- Identifies exact matches
- Lists duplicate files

**Output**: List of duplicate entries

**When to Use**: To ensure dataset diversity

---

#### 5. 📊 Character Count Analysis
**Purpose**: Analyzes character count distribution

**Provides**:
- Average characters per sample
- Minimum/Maximum lengths
- Standard deviation
- Median length

**When to Use**: To understand text length distribution

---

### Category 2: Text Analysis (Functions 6-8)

#### 6. 📈 Word Frequency Analysis
**Purpose**: Finds most common words in dataset

**Parameters**:
- `top_n`: Number of top words to display (Default: 20)

**Output**: List of most frequent words with counts

**When to Use**: To understand vocabulary distribution

---

#### 7. 🗑️ Remove Short Samples
**Purpose**: Filters out samples with too few words

**Parameters**:
- `min_words`: Minimum words required (Default: 5)

**Output**: `metadata_filtered.csv`

**When to Use**: To remove low-quality short samples

---

#### 8. 📋 Generate Dataset Report
**Purpose**: Creates comprehensive dataset report

**Includes**:
- Total samples, words, characters
- Average statistics
- Dataset readiness assessment
- Recommendations

**Output**: `dataset_report.txt`

**When to Use**: For complete dataset overview

---

### Category 3: Dataset Management (Functions 9-14)

#### 9. 🔗 Merge Datasets
**Purpose**: Combines multiple metadata files

**Merges**:
- `metadata.csv`
- `metadata_balanced.csv`
- `metadata_filtered.csv`

**Features**:
- Removes duplicates automatically
- Combines all unique entries

**Output**: `metadata_merged.csv`

---

#### 10. 📦 Export to JSON
**Purpose**: Exports metadata to JSON format

**JSON Structure**:
```json
{
  "audio_file": "chunk_0001.wav",
  "transcription": "text here",
  "word_count": 10,
  "char_count": 50
}
```

**Output**: `metadata.json`

**When to Use**: For JSON-based training pipelines

---

#### 11. 🎲 Random Sample
**Purpose**: Extracts random samples from dataset

**Parameters**:
- `num_samples`: Number of samples (Default: 20)

**Output**: `sample_dataset.csv` with preview

**When to Use**: For testing or creating demo datasets

---

#### 12. 💾 Calculate Dataset Size
**Purpose**: Calculates total disk space used

**Provides**:
- Total file count
- Total size (MB/GB)
- Average file size
- Disk space assessment

**When to Use**: To check storage requirements

---

#### 13. 🧹 Clean Invalid Entries
**Purpose**: Removes entries with corrupted/invalid audio

**Checks**:
- Audio file can be loaded
- Audio duration > 0.5 seconds
- File integrity

**Output**: `metadata_cleaned.csv`

**When to Use**: After detecting audio issues

---

#### 14. 💾 Create Backup
**Purpose**: Creates timestamped backup of entire dataset

**Backup Includes**:
- All audio chunks
- All metadata files
- Statistics and reports

**Output**: `tts_training_data_backup_YYYYMMDD_HHMMSS/`

**When to Use**: Before major changes or cleaning

---

### Category 4: Audio Processing (Functions 15-17)

#### 15. 🔊 Normalize All Audio
**Purpose**: Normalizes volume of all audio files

**Process**:
- Loads each audio file
- Applies normalization
- Overwrites with normalized version

**When to Use**: For inconsistent audio volumes

---

#### 16. 📊 Generate Statistics Chart
**Purpose**: Creates visual charts of dataset statistics

**Charts**:
1. Duration Distribution histogram
2. Word Count Distribution histogram

**Output**: `statistics/statistics_chart.png`

**When to Use**: For visual data analysis

---

#### 17. 🔎 Find Long Samples
**Purpose**: Identifies samples longer than specified duration

**Parameters**:
- `max_duration`: Maximum duration threshold (Default: 20s)

**Output**: List of long samples with durations

**When to Use**: To identify outliers

---

### Category 5: Advanced Filtering (Functions 18-20)

#### 18. 🔍 Text Length Filter
**Purpose**: Filters samples by character count range

**Parameters**:
- `min_chars`: Minimum characters (Default: 50)
- `max_chars`: Maximum characters (Default: 500)

**Output**: `metadata_length_filtered.csv`

**When to Use**: For specific text length requirements

---

#### 19. 📊 3-Way Split (Train/Val/Test)
**Purpose**: Creates three-way dataset split

**Parameters**:
- `train_pct`: Training percentage (Default: 70%)
- `val_pct`: Validation percentage (Default: 15%)
- `test_pct`: Test percentage (Default: 15%)

**Output**:
- `train.txt`
- `val.txt`
- `test.txt`

**When to Use**: For complete model evaluation setup

---

#### 20. 📄 Export Dataset Summary
**Purpose**: Exports comprehensive JSON summary

**Summary Includes**:
- Total samples, duration, words, characters
- Average statistics
- Generation timestamp
- File locations

**Output**: `dataset_summary.json`

**When to Use**: For dataset documentation

---

## 📁 Output Files

### Folder Structure
```
tts_training_data/
├── audio_chunks/           # All processed audio chunks
│   ├── chunk_0000.wav
│   ├── chunk_0001.wav
│   └── ...
├── filtered_chunks/        # Quality-filtered chunks only
│   ├── chunk_0000.wav
│   ├── chunk_0005.wav
│   └── ...
├── statistics/             # Statistics and reports
│   ├── audio_statistics.json
│   ├── analysis_report.txt
│   └── statistics_chart.png
├── transcriptions.txt      # All transcriptions
├── metadata.csv           # Main metadata file
├── train.txt              # Training set
├── val.txt                # Validation set
├── test.txt               # Test set
└── dataset_summary.json   # Complete summary
```

### File Formats

#### transcriptions.txt
```
chunk_0000.wav|यह एक उदाहरण है
chunk_0001.wav|This is an example
chunk_0002.wav|[UNINTELLIGIBLE]
```

#### metadata.csv
```
chunk_0000.wav|यह एक उदाहरण है
chunk_0001.wav|This is an example
```
*Note: Only valid transcriptions, no errors*

#### audio_statistics.json
```json
[
  {
    "filename": "chunk_0000.wav",
    "duration": 8.5,
    "transcription": "यह एक उदाहरण है",
    "word_count": 4,
    "rms_energy": 0.0234,
    "zero_crossing_rate": 0.0456,
    "spectral_centroid": 2456.78
  }
]
```

#### dataset_summary.json
```json
{
  "total_samples": 150,
  "total_duration": 1250.5,
  "total_words": 1500,
  "total_characters": 7500,
  "avg_duration": 8.34,
  "avg_words": 10.0,
  "avg_characters": 50.0,
  "generated_date": "2025-10-20 14:30:00"
}
```

---

## 🎯 Workflow Guide

### Recommended Workflow for TTS Training

#### Step 1: Process Audio
1. Upload your audio file
2. Select language (Hindi/English)
3. Enable audio enhancements (normalization, silence removal)
4. Set appropriate chunk duration (10s recommended)
5. Click "Generate & Process"
6. Wait for completion

#### Step 2: Validate Dataset
1. Go to "Dataset Validation" tab
2. Click "Validate Dataset"
3. Review any issues found
4. Fix issues if necessary

#### Step 3: Clean & Filter
1. Use "Detect Duplicates" to find duplicates
2. Use "Clean Invalid Entries" to remove corrupted files
3. Use "Remove Short Samples" to filter quality
4. Use "Text Length Filter" if needed

#### Step 4: Balance Dataset
1. Go to "Balance Dataset" tab
2. Set minimum samples per word (5 recommended)
3. Click "Balance Dataset"
4. Review balanced results

#### Step 5: Create Backup
1. Go to "Create Backup" tab
2. Click "Create Backup"
3. Save backup location

#### Step 6: Split Dataset
1. Go to "3-Way Split" tab
2. Set percentages (70/15/15 recommended)
3. Click "Split"
4. Download train.txt, val.txt, test.txt

#### Step 7: Generate Reports
1. Generate "Dataset Report" for overview
2. Generate "Statistics Chart" for visuals
3. Export "Dataset Summary" for documentation

#### Step 8: Start Training
1. Use `train.txt` for training
2. Use `val.txt` for validation
3. Use `test.txt` for final evaluation
4. Audio files are in `filtered_chunks/` folder

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "No module named 'gradio'"
**Solution**: Install required packages
```bash
pip install gradio pydub SpeechRecognition librosa soundfile matplotlib
```

#### 2. "FFmpeg not found"
**Solution**: Install FFmpeg and add to PATH
- Windows: Download and add to system PATH
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

#### 3. "Speech recognition failed"
**Solution**: 
- Check internet connection (Google API requires internet)
- Ensure audio quality is good
- Try enabling noise reduction
- Check if correct language is selected

#### 4. "[UNINTELLIGIBLE]" in transcriptions
**Causes**:
- Poor audio quality
- Background noise
- Unclear speech
- Wrong language selected

**Solutions**:
- Use clearer audio
- Enable noise reduction
- Check language setting
- Manually review and edit

#### 5. "Out of memory" error
**Solution**:
- Process smaller audio files
- Close other applications
- Increase system RAM
- Reduce chunk duration

#### 6. Very slow processing
**Causes**:
- Large audio file
- Slow internet connection
- Too many chunks

**Solutions**:
- Split large files manually
- Check internet speed
- Increase chunk duration
- Process during off-peak hours

#### 7. Empty output folder
**Solution**:
- Check if audio file was uploaded correctly
- Verify file format is supported
- Check console for error messages
- Try different audio file

---

## 🎬 YouTube Channel

### Mind Hack Secrets
**Channel**: [Mind Hack Secrets](https://www.youtube.com/@MindHackSecrets-o4y?sub_confirmation=1)

**Content**:
- AI और TTS Tutorials
- Machine Learning Projects
- Technology Tips & Tricks
- Hindi & English Content

**Subscribe for**:
- TTS Model Training Tutorials
- AI Tools & Applications
- Python Programming
- Dataset Preparation Tips

**Click the RED SUBSCRIBE BUTTON in the app** to directly subscribe with one click!

---

## 📊 Technical Specifications

### Audio Processing
- **Sample Rate**: 22,050 Hz (standard for TTS)
- **Channels**: Mono (1 channel)
- **Format**: WAV (lossless)
- **Bit Depth**: 16-bit

### Speech Recognition
- **Engine**: Google Speech Recognition API
- **Languages**: Hindi (hi-IN), English (en-US)
- **Requires**: Internet connection

### Noise Reduction Algorithm
- **Method**: Spectral Gating
- **Parameters**: 
  - n_fft: 2048
  - hop_length: 512
  - Noise floor: 10th percentile
  - Threshold: 1.5x noise floor

---

## 🔐 Privacy & Security

### Data Privacy
- All processing is done **locally** on your machine
- Audio files are **not uploaded** to any server
- Only transcription uses Google API (audio is processed locally, text is sent)
- No data is stored or shared

### Recommendations
- Keep audio files secure
- Backup important datasets
- Don't process sensitive audio without permission
- Follow copyright laws for audio content

---

## 📝 Tips & Best Practices

### Audio Quality
✅ **Do**:
- Use clear, high-quality audio
- Ensure minimal background noise
- Use consistent speaking pace
- Record in quiet environment
- Use good microphone

❌ **Don't**:
- Use low-bitrate compressed audio
- Process audio with music/multiple speakers
- Use audio with echo or reverb
- Mix different audio qualities

### Dataset Preparation
✅ **Do**:
- Validate dataset before training
- Remove duplicates and errors
- Balance word distribution
- Split into train/val/test
- Create backups regularly

❌ **Don't**:
- Skip validation steps
- Train on imbalanced data
- Ignore quality issues
- Mix different languages/speakers

### TTS Training
✅ **Do**:
- Use 70/15/15 or 80/10/10 split
- Ensure consistent audio quality
- Have diverse vocabulary
- Check for proper transcriptions
- Use filtered chunks

❌ **Don't**:
- Use unvalidated data
- Train on very short samples
- Ignore audio artifacts
- Skip test set evaluation

---

## 🆘 Support & Contact

### Getting Help
1. **Check this documentation first**
2. **Review Troubleshooting section**
3. **Watch YouTube tutorials**: [Mind Hack Secrets](https://www.youtube.com/@MindHackSecrets-o4y)
4. **Check Gradio documentation**: https://gradio.app/docs/

### Feature Requests
If you need additional features, please:
1. Subscribe to Mind Hack Secrets channel
2. Leave a comment on relevant videos
3. Describe your use case clearly

---

## 📄 License & Credits

### Tool Credits
- **Created by**: Mind Hack Secrets
- **YouTube**: [@MindHackSecrets-o4y](https://www.youtube.com/@MindHackSecrets-o4y)

### Libraries Used
- **Gradio**: UI Framework
- **pydub**: Audio processing
- **librosa**: Audio analysis
- **SpeechRecognition**: Transcription
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization

---

## 🎓 Learning Resources

### Recommended Tutorials
1. **TTS Model Training Basics** - Mind Hack Secrets YouTube
2. **Dataset Preparation Guide** - Mind Hack Secrets YouTube
3. **Audio Processing Tips** - Mind Hack Secrets YouTube

### Further Reading
- Gradio Documentation
- Librosa Tutorials
- TTS Model Papers
- Speech Recognition Guides

---

## 🚀 Future Updates

### Coming Soon
- Multi-speaker support
- Custom noise profiles
- Advanced augmentation
- Real-time preview
- Batch file processing
- More language support

**Stay tuned on Mind Hack Secrets YouTube channel for updates!**

---

## ✅ Changelog

### Version 1.0.0 (Current)
- Initial release
- 20 AI training functions
- Hindi + English support
- Bilingual interface
- YouTube channel integration
- Complete documentation

---

## 🙏 Acknowledgments

Thank you for using this tool! 

**Don't forget to**:
- ⭐ Subscribe to Mind Hack Secrets
- 👍 Like helpful tutorials
- 💬 Share your feedback
- 📢 Share with others

**Happy TTS Training! 🎙️✨**

---

*Last Updated: October 2025*  
*Documentation Version: 1.0.0*

