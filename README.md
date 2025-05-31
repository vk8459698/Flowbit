# Multi-Agent AI Document Processing System

A sophisticated Python-based document processing system that uses multiple AI agents to classify, analyze, and extract information from various document formats including PDFs, JSON files, emails, and plain text.

## 🎯 Features

- **Multi-Agent Architecture**: Specialized agents for different document types and processing tasks
- **AI-Powered Classification**: Uses Groq's LLaMA model for intelligent document intent classification
- **Multiple Format Support**: PDF, JSON, Email, and plain text processing
- **Persistent Memory**: SQLite database for storing processing history and context
- **Thread-Based Context**: Link related documents together for contextual processing
- **Interactive & Batch Processing**: Command-line interface with multiple operation modes

## 📋 Document Classification

### Supported Formats
- **PDF**: Extracts text content from PDF documents
- **JSON**: Parses and validates JSON structure with anomaly detection
- **EMAIL**: Extracts headers, urgency levels, and key information
- **TEXT**: General text document processing

### Intent Classification
- **INVOICE**: Bills, payment requests, financial documents
- **RFQ**: Request for quotes, procurement documents
- **COMPLAINT**: Customer complaints, issues, problems
- **REGULATION**: Legal documents, compliance, policies
- **CONTRACT**: Agreements, terms and conditions
- **REPORT**: Analysis, summaries, findings
- **GENERAL**: Everything else

## 🛠️ Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Dependencies
```
groq
PyPDF2
sqlite3 (built-in)
uuid (built-in)
datetime (built-in)
json (built-in)
re (built-in)
os (built-in)
argparse (built-in)
dataclasses (built-in)
enum (built-in)
typing (built-in)
io (built-in)
```

### API Key Setup
1. Get your Groq API key from [Groq Console](https://console.groq.com/)
2. Replace the `GROQ_API_KEY` variable in the code with your actual API key
3. **Security Note**: For production use, store the API key as an environment variable

## 🚀 Usage

### Command Line Interface

#### Process a File
```bash
python script.py --file document.pdf
python script.py -f invoice.json
```

#### Process Text Directly
```bash
python script.py --text "Your document content here"
python script.py -t "Email content to analyze"
```

#### Interactive Mode
```bash
python script.py --interactive
python script.py -i
```

#### View Processing History
```bash
python script.py --history
```

#### Advanced Options
```bash
# Process with thread context
python script.py --file document.pdf --thread "project-alpha"

# Save to custom output file
python script.py --file document.pdf --output "results.json"

# Combine multiple options
python script.py --file contract.pdf --thread "legal-docs" --output "contract_analysis.json"
```

## 🏗️ Architecture

### Core Components

#### 1. SharedMemoryModule
- SQLite database for persistent storage
- Stores processing results with full context
- Enables thread-based document linking
- Maintains processing history

#### 2. ClassifierAgent
- Primary document classification
- Format detection (PDF, JSON, Email, Text)
- Intent classification using AI
- PDF text extraction

#### 3. JSONAgent
- JSON structure validation
- Field analysis and anomaly detection
- Missing field identification
- Business document field checking

#### 4. EmailAgent
- Email header extraction (From, To, Subject)
- Urgency level assessment
- Key phrase extraction using AI
- Action item identification

#### 5. MultiAgentOrchestrator
- Coordinates all agents
- Routes documents to appropriate processors
- Combines results from multiple agents
- Manages workflow execution

## 📊 Output Format

The system generates comprehensive JSON output with two main sections:

### Classification Results
```json
{
  "classification": {
    "id": "unique-document-id",
    "format": "PDF|JSON|EMAIL|TEXT",
    "intent": "INVOICE|RFQ|COMPLAINT|etc",
    "confidence": 0.85,
    "timestamp": "2024-01-15T10:30:00",
    "source": "filename.pdf"
  }
}
```

### Processing Results
```json
{
  "processing": {
    "type": "email_alert",
    "extracted_fields": {
      "sender": "john.customer@email.com",
      "subject": "URGENT: Critical System Failure - Production Down",
      "urgency": "high",
      "recipients": [
        "support@techcorp.com",
        "manager@techcorp.com"
      ]
    },
    "anomalies": [],
    "key_information": {}
  }
}

```

## 🔧 Configuration

### AI Model Settings
- **Model**: `llama-3.1-8b-instant` (Groq)
- **Temperature**: 0.1 (low randomness for consistent classification)
- **Max Tokens**: Varies by task (50-300)

### Database Configuration
- **Default Path**: `agent_memory.db`
- **Storage**: All processing results and metadata
- **Schema**: Optimized for fast retrieval and context linking

## 🎮 Interactive Mode Features

When running in interactive mode (`--interactive`), you can:
- Process multiple documents in sequence
- Maintain thread context across inputs
- View immediate formatted results
- Save results automatically

Example session:
```
Multi-Agent AI Document Processing System
==================================================
Interactive Mode - Type 'quit' to exit

Enter text to process (or 'quit'): From: john@company.com
Subject: Urgent Payment Required
...

============================================================
MULTI-AGENT PROCESSING RESULT
============================================================
Document ID: abc-123-def
Format: EMAIL
Intent: INVOICE
Confidence: 0.92
...
```

## 📈 Processing History

The system maintains a complete history of all processed documents:
- Unique document IDs
- Processing timestamps
- Classification results
- Confidence scores
- Thread associations

Access history with:
```bash
python script.py --history
```

## 🔍 Advanced Features

### Thread-Based Context
Link related documents together:
```bash
python script.py --file doc1.pdf --thread "project-alpha"
python script.py --file doc2.json --thread "project-alpha"
```

### Confidence Scoring
Each classification includes a confidence score (0.0-1.0) indicating the AI's certainty in its classification.

### Fallback Classification
If AI classification fails, the system uses keyword-based fallback classification to ensure robust operation.

### Error Handling
Comprehensive error handling for:
- Invalid file formats
- API failures
- Database connection issues
- Malformed content

## 🛡️ Security Considerations

- **API Key Management**: Store API keys securely as environment variables
- **Data Privacy**: All processing is done locally except for AI classification calls
- **Database Security**: SQLite database stores extracted content - ensure proper file permissions

## 🔄 Extensibility

The modular architecture allows easy extension:
- Add new document formats by creating new agents
- Extend classification intents by updating the enum
- Add new processing capabilities to existing agents
- Integrate additional AI models or services

## 🐛 Troubleshooting

### Common Issues

**PDF Processing Errors**
- Ensure PyPDF2 is installed correctly
- Check file permissions
- Verify PDF is not password-protected

**API Connection Issues**
- Verify Groq API key is valid
- Check internet connectivity
- Monitor API rate limits

**Database Errors**
- Ensure write permissions in current directory
- Check available disk space
- Verify SQLite installation

## 📝 Example Use Cases

1. **Invoice Processing**: Automatically classify and extract data from invoices
2. **Email Triage**: Analyze email urgency and extract action items
3. **Document Management**: Organize large document collections by type and intent
4. **Compliance Monitoring**: Identify regulatory documents and requirements
5. **Contract Analysis**: Process legal agreements and extract key terms

## 🤝 Contributing

The system is designed for easy extension and modification. Key areas for contribution:
- Additional document format support
- Enhanced AI prompts for better classification
- Performance optimizations
- Additional output formats
- Integration with other AI services

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the command-line help: `python script.py --help`
3. Examine the SQLite database for processing history
4. Enable verbose logging for debugging

---

*Built with ❤️ using Python, Groq AI, and multi-agent architecture principles*
