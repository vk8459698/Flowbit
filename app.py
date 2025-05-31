import json
import sqlite3
import datetime
import uuid
import os
import re
import PyPDF2
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import io
from groq import Groq
import argparse
import sys

# Initialize Groq client with API key
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# Enums for classification
class DocumentFormat(Enum):
    PDF = "PDF"
    JSON = "JSON"
    EMAIL = "EMAIL"
    TEXT = "TEXT"

class DocumentIntent(Enum):
    INVOICE = "INVOICE"
    RFQ = "RFQ"
    COMPLAINT = "COMPLAINT"
    REGULATION = "REGULATION"
    GENERAL = "GENERAL"
    CONTRACT = "CONTRACT"
    REPORT = "REPORT"

@dataclass
class ProcessingResult:
    id: str
    format: DocumentFormat
    intent: DocumentIntent
    extracted_data: Dict[str, Any]
    timestamp: datetime.datetime
    source: str
    confidence: float
    thread_id: Optional[str] = None

class SharedMemoryModule:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_history (
                id TEXT PRIMARY KEY,
                format TEXT,
                intent TEXT,
                extracted_data TEXT,
                timestamp TEXT,
                source TEXT,
                confidence REAL,
                thread_id TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_result(self, result: ProcessingResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO processing_history 
            (id, format, intent, extracted_data, timestamp, source, confidence, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.id,
            result.format.value,
            result.intent.value,
            json.dumps(result.extracted_data),
            result.timestamp.isoformat(),
            result.source,
            result.confidence,
            result.thread_id
        ))
        conn.commit()
        conn.close()
    
    def get_thread_context(self, thread_id: str) -> List[ProcessingResult]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM processing_history 
            WHERE thread_id = ? 
            ORDER BY timestamp DESC
        ''', (thread_id,))
        results = []
        for row in cursor.fetchall():
            result = ProcessingResult(
                id=row[0],
                format=DocumentFormat(row[1]),
                intent=DocumentIntent(row[2]),
                extracted_data=json.loads(row[3]),
                timestamp=datetime.datetime.fromisoformat(row[4]),
                source=row[5],
                confidence=row[6],
                thread_id=row[7]
            )
            results.append(result)
        conn.close()
        return results
    
    def get_recent_history(self, limit: int = 10) -> List[ProcessingResult]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM processing_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        results = []
        for row in cursor.fetchall():
            result = ProcessingResult(
                id=row[0],
                format=DocumentFormat(row[1]),
                intent=DocumentIntent(row[2]),
                extracted_data=json.loads(row[3]),
                timestamp=datetime.datetime.fromisoformat(row[4]),
                source=row[5],
                confidence=row[6],
                thread_id=row[7]
            )
            results.append(result)
        conn.close()
        return results

class ClassifierAgent:
    def __init__(self, memory: SharedMemoryModule):
        self.memory = memory
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    def classify_format(self, content: str, filename: str = "") -> DocumentFormat:
        if filename.lower().endswith('.pdf'):
            return DocumentFormat.PDF
        
        # Try to parse as JSON
        try:
            json.loads(content)
            return DocumentFormat.JSON
        except:
            pass
        
        # Check for email patterns
        email_patterns = [
            r'From:.*?To:.*?Subject:',
            r'@.*?\.(com|org|net|edu)',
            r'Subject:.*?',
            r'Dear.*?,'
        ]
        
        for pattern in email_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                return DocumentFormat.EMAIL
        
        return DocumentFormat.TEXT  # Default fallback
    
    def classify_intent(self, content: str, format_type: DocumentFormat) -> Tuple[DocumentIntent, float]:
        # LLM-based classification
        try:
            prompt = f"""
            Analyze the following document and classify its intent. 
            Content: {content[:1000]}...
            
            Classify the intent as one of:
            - INVOICE: Bills, payment requests, financial documents
            - RFQ: Request for quotes, procurement documents
            - COMPLAINT: Customer complaints, issues, problems
            - REGULATION: Legal documents, compliance, policies
            - CONTRACT: Agreements, terms and conditions
            - REPORT: Analysis, summaries, findings
            - GENERAL: Everything else
            
            Respond with only the classification and confidence (0-1):
            Format: INTENT|CONFIDENCE
            """
            
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            parts = result.split('|')
            
            if len(parts) == 2:
                intent_str, confidence_str = parts
                intent = DocumentIntent(intent_str.strip())
                confidence = float(confidence_str.strip())
                return intent, confidence
            else:
                return DocumentIntent.GENERAL, 0.5
                
        except Exception as e:
            print(f"LLM classification failed: {e}")
            # Fallback classification without LLM
            keywords = {
                DocumentIntent.INVOICE: ['invoice', 'bill', 'payment', 'amount due', 'total', '$'],
                DocumentIntent.RFQ: ['request for quote', 'rfq', 'quotation', 'bid', 'proposal'],
                DocumentIntent.COMPLAINT: ['complaint', 'issue', 'problem', 'dissatisfied', 'unhappy'],
                DocumentIntent.REGULATION: ['regulation', 'compliance', 'policy', 'legal', 'requirement'],
                DocumentIntent.CONTRACT: ['contract', 'agreement', 'terms', 'conditions', 'signature'],
                DocumentIntent.REPORT: ['report', 'analysis', 'summary', 'findings', 'conclusion']
            }
            
            content_lower = content.lower()
            scores = {}
            
            for intent, intent_keywords in keywords.items():
                score = sum(1 for keyword in intent_keywords if keyword in content_lower)
                scores[intent] = score
            
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent] / len(keywords[best_intent]) if scores[best_intent] > 0 else 0.3
            
            return best_intent, min(confidence, 1.0)
    
    def process(self, content: str, filename: str = "", thread_id: Optional[str] = None) -> ProcessingResult:
        # Determine format
        doc_format = self.classify_format(content, filename)
        
        # Classify intent
        intent, confidence = self.classify_intent(content, doc_format)
        
        # Create result
        result = ProcessingResult(
            id=str(uuid.uuid4()),
            format=doc_format,
            intent=intent,
            extracted_data={"raw_content": content[:500]},  # Store first 500 chars
            timestamp=datetime.datetime.now(),
            source=filename or "direct_input",
            confidence=confidence,
            thread_id=thread_id
        )
        
        # Store in memory
        self.memory.store_result(result)
        
        return result

class JSONAgent:
    def __init__(self, memory: SharedMemoryModule):
        self.memory = memory
    
    def process(self, json_content: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            data = json.loads(json_content)
            
            # Extract common fields
            extracted = {
                "type": "json_processing",
                "fields_count": len(data) if isinstance(data, dict) else 0,
                "structure": type(data).__name__,
                "anomalies": [],
                "missing_fields": []
            }
            
            # Check for common business document fields
            expected_fields = ['id', 'date', 'amount', 'customer', 'vendor', 'description']
            if isinstance(data, dict):
                for field in expected_fields:
                    if field not in data:
                        extracted["missing_fields"].append(field)
                
                # Check for anomalies
                for key, value in data.items():
                    if value is None or value == "":
                        extracted["anomalies"].append(f"Empty value for {key}")
            
            # Extract specific data based on structure
            if isinstance(data, dict):
                extracted["extracted_fields"] = {}
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool)):
                        extracted["extracted_fields"][key] = value
            
            return extracted
            
        except json.JSONDecodeError as e:
            return {
                "type": "json_processing",
                "error": f"Invalid JSON: {str(e)}",
                "anomalies": ["Invalid JSON format"],
                "missing_fields": ["all"]
            }

class EmailAgent:
    def __init__(self, memory: SharedMemoryModule):
        self.memory = memory
    
    def extract_email_info(self, email_content: str) -> Dict[str, Any]:
        # Extract email headers and content
        email_data = {
            "sender": None,
            "subject": None,
            "urgency": "medium",
            "intent": "general",
            "key_phrases": [],
            "recipients": [],
            "timestamp": None
        }
        
        # Extract sender
        sender_match = re.search(r'From:\s*([^\n\r]+)', email_content, re.IGNORECASE)
        if sender_match:
            email_data["sender"] = sender_match.group(1).strip()
        
        # Extract subject
        subject_match = re.search(r'Subject:\s*([^\n\r]+)', email_content, re.IGNORECASE)
        if subject_match:
            email_data["subject"] = subject_match.group(1).strip()
        
        # Extract recipients
        to_match = re.search(r'To:\s*([^\n\r]+)', email_content, re.IGNORECASE)
        if to_match:
            email_data["recipients"] = [r.strip() for r in to_match.group(1).split(',')]
        
        # Determine urgency
        urgency_keywords = {
            "high": ["urgent", "asap", "immediate", "critical", "emergency"],
            "medium": ["important", "priority", "soon"],
            "low": ["fyi", "when convenient", "no rush"]
        }
        
        content_lower = email_content.lower()
        for level, keywords in urgency_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                email_data["urgency"] = level
                break
        
        # Extract key phrases using LLM
        try:
            prompt = f"""
            Extract key information from this email:
            {email_content[:800]}
            
            Provide:
            1. Main purpose/intent
            2. Key action items
            3. Important dates or deadlines
            4. Key entities mentioned
            
            Format as JSON with keys: intent, action_items, dates, entities
            """
            
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=300
            )
            
            llm_result = response.choices[0].message.content.strip()
            try:
                llm_data = json.loads(llm_result)
                email_data.update(llm_data)
            except:
                email_data["llm_analysis"] = llm_result
                
        except Exception as e:
            email_data["llm_error"] = str(e)
        
        return email_data
    
    def process(self, email_content: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        return self.extract_email_info(email_content)

class MultiAgentOrchestrator:
    def __init__(self):
        self.memory = SharedMemoryModule()
        self.classifier = ClassifierAgent(self.memory)
        self.json_agent = JSONAgent(self.memory)
        self.email_agent = EmailAgent(self.memory)
    
    def process_document(self, content: str, filename: str = "", thread_id: Optional[str] = None) -> Dict[str, Any]:
        # Step 1: Classify the document
        classification_result = self.classifier.process(content, filename, thread_id)
        
        # Step 2: Route to appropriate agent
        agent_result = {}
        
        if classification_result.format == DocumentFormat.JSON:
            agent_result = self.json_agent.process(content, thread_id)
        elif classification_result.format == DocumentFormat.EMAIL:
            agent_result = self.email_agent.process(content, thread_id)
        elif classification_result.format == DocumentFormat.PDF:
            # For PDF, we already extracted text in classifier
            agent_result = {
                "type": "pdf_processing",
                "text_length": len(content),
                "pages_estimated": len(content) // 2000,  # Rough estimate
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
        else:  # TEXT
            agent_result = {
                "type": "text_processing",
                "text_length": len(content),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
        
        # Step 3: Combine results
        final_result = {
            "classification": {
                "id": classification_result.id,
                "format": classification_result.format.value,
                "intent": classification_result.intent.value,
                "confidence": classification_result.confidence,
                "timestamp": classification_result.timestamp.isoformat(),
                "source": classification_result.source
            },
            "processing": agent_result,
            "thread_id": thread_id or classification_result.id
        }
        
        return final_result
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        results = self.memory.get_recent_history(limit)
        return [
            {
                "id": r.id,
                "format": r.format.value,
                "intent": r.intent.value,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "source": r.source,
                "confidence": r.confidence,
                "preview": str(r.extracted_data)[:100] + "..."
            }
            for r in results
        ]

def save_result_to_file(result: Dict[str, Any], output_file: str = None):
    """Save processing result to a JSON file"""
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"processing_result_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return output_file

def print_result_summary(result: Dict[str, Any]):
    """Print a formatted summary of the processing result"""
    print("\n" + "="*60)
    print("MULTI-AGENT PROCESSING RESULT")
    print("="*60)
    
    classification = result["classification"]
    processing = result["processing"]
    
    print(f"Document ID: {classification['id']}")
    print(f"Format: {classification['format']}")
    print(f"Intent: {classification['intent']}")
    print(f"Confidence: {classification['confidence']:.2f}")
    print(f"Timestamp: {classification['timestamp']}")
    print(f"Source: {classification['source']}")
    print(f"Thread ID: {result['thread_id']}")
    
    print("\nProcessing Details:")
    print("-" * 30)
    
    if processing.get('type') == 'json_processing':
        print(f"• JSON Fields: {processing.get('fields_count', 0)}")
        print(f"• Structure: {processing.get('structure', 'Unknown')}")
        if processing.get('anomalies'):
            print(f"• Anomalies: {', '.join(processing['anomalies'])}")
        if processing.get('missing_fields'):
            print(f"• Missing Fields: {', '.join(processing['missing_fields'])}")
    
    elif processing.get('type') in ['pdf_processing', 'text_processing']:
        print(f"• Text Length: {processing.get('text_length', 0)} characters")
        if processing.get('pages_estimated'):
            print(f"• Estimated Pages: {processing['pages_estimated']}")
        print(f"• Preview: {processing.get('content_preview', 'N/A')}")
    
    elif 'sender' in processing:  # Email processing
        print(f"• Sender: {processing.get('sender', 'Unknown')}")
        print(f"• Subject: {processing.get('subject', 'No subject')}")
        print(f"• Urgency: {processing.get('urgency', 'medium')}")
        if processing.get('recipients'):
            print(f"• Recipients: {', '.join(processing['recipients'])}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Document Processing System')
    parser.add_argument('--file', '-f', type=str, help='Path to document file to process')
    parser.add_argument('--text', '-t', type=str, help='Text content to process directly')
    parser.add_argument('--thread', type=str, help='Thread ID for linking related documents')
    parser.add_argument('--output', '-o', type=str, help='Output file path for results')
    parser.add_argument('--history', action='store_true', help='Show processing history')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    print("Multi-Agent AI Document Processing System")
    print("=" * 50)
    print("Groq API initialized successfully")
    print("Database initialized successfully")
    print()
    
    if args.history:
        print("Recent Processing History:")
        print("-" * 30)
        history = orchestrator.get_processing_history()
        for item in history:
            print(f"• {item['timestamp']} | {item['format']} | {item['intent']} | {item['source']}")
        return
    
    if args.interactive:
        print("Interactive Mode - Type 'quit' to exit")
        while True:
            try:
                user_input = input("\nEnter text to process (or 'quit'): ").strip()
                if user_input.lower() == 'quit':
                    break
                if user_input:
                    result = orchestrator.process_document(user_input, "interactive_input", args.thread)
                    print_result_summary(result)
                    
                    save_file = save_result_to_file(result, args.output)
                    print(f"Result saved to: {save_file}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
        return
    
    if args.file:
        print(f"Processing file: {args.file}")
        
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found")
            return
        
        try:
            if args.file.lower().endswith('.pdf'):
                content = orchestrator.classifier.extract_pdf_text(args.file)
            else:
                with open(args.file, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            result = orchestrator.process_document(content, args.file, args.thread)
            print_result_summary(result)
            
            save_file = save_result_to_file(result, args.output)
            print(f"Result saved to: {save_file}")
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
    
    elif args.text:
        print("Processing text input...")
        
        result = orchestrator.process_document(args.text, "command_line_input", args.thread)
        print_result_summary(result)
        
        save_file = save_result_to_file(result, args.output)
        print(f"Result saved to: {save_file}")
    
    else:
        print("❓ No input provided. Use --help for usage information")
        print("\nQuick examples:")
        print("  python script.py --file document.pdf")
        print("  python script.py --text 'Your text here'")
        print("  python script.py --interactive")
        print("  python script.py --history")

if __name__ == "__main__":
    main()