"""
Privacy Detection and Redaction
===============================
This module is the "Secret Service" for your data.
It scans text for sensitive info (like phone numbers) and blacks it out (<REDACTED>) so it's safe to share.
"""

from typing import List, Dict, Optional
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PrivacyGuard:
    """
    The main Privacy Firewall.
    
    It uses Microsoft Presidio (a powerful library) to catch things like:
    - Phone Numbers
    - Credit Cards
    - Emails
    
    If it sees "My number is 555-0199", it changes it to "My number is <REDACTED>" (or similar),
    unless you are the admin or the owner.
    """
    
    def __init__(self, language: str = 'en', config_path: str = "configs/policy.yaml"):
        """
        Setup the privacy engines.
        """
        # First, check if privacy is even turned on in the config file.
        self.enabled = True
        try:
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                self.enabled = cfg.get('privacy', {}).get('enabled', True)
        except Exception:
            pass # If config is missing, default to Safe/Enabled.
            
        # Spin up the specialized engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.language = language
        
        # Here's the list of stuff we look for by default
        self.default_entities = [
            "PHONE_NUMBER",
            "EMAIL_ADDRESS", 
            "CREDIT_CARD",
            "US_SSN",
            "PERSON",       # Names
            "LOCATION",     # Cities, Addresses
            "DATE_TIME",
            "IP_ADDRESS",
            "URL"
        ]
    
    def sanitize(
        self, 
        text: str,
        entities: Optional[List[str]] = None,
        operators: Optional[Dict[str, OperatorConfig]] = None,
        role: str = "user"
    ) -> str:
        """
        The main function to clean text.
        Pass in a string, get back a safe string.
        """
        # 1. Is the system turned off?
        if not self.enabled:
            return text

        # 2. Is this an Admin? They get to see everything.
        if role == "admin":
            return text
        
        # Use defaults if you didn't ask for specific checks
        if entities is None:
            entities = self.default_entities
        
        # Step 1: Scan the text for secrets
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=entities,
            language=self.language
        )
        
        # Step 2: Decide HOW to hide them (Mask vs Redact)
        if operators is None:
            operators = self._get_default_operators()
        
        # Step 3: Perform the scrubbing
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators
        )
        
        return anonymized_result.text
    
    def _get_default_operators(self) -> Dict[str, OperatorConfig]:
        """
        Defines how we hide different secrets.
        
        - Regular stuff: Replace with <REDACTED>
        - Phones/Cards: Show the last few digits (like ****-1234) so users recognize them.
        """
        return {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PHONE_NUMBER": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "*",
                    "chars_to_mask": 7,
                    "from_end": True
                }
            ),
            "CREDIT_CARD": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "*",
                    "chars_to_mask": 12,
                    "from_end": True
                }
            )
        }
    
    def add_custom_recognizer(
        self,
        entity_name: str,
        patterns: List[str],
        scores: Optional[List[float]] = None
    ) -> None:
        """
        Teach the guard to recognize new secrets.
        
        Example: "Project IDs look like PRJ-1234"
        """
        if scores is None:
            scores = [0.85] * len(patterns)
        
        # Convert simple regex strings into complex Pattern objects
        pattern_objects = [
            Pattern(
                name=f"{entity_name.lower()}_pattern_{i}",
                regex=pattern,
                score=score
            )
            for i, (pattern, score) in enumerate(zip(patterns, scores))
        ]
        
        # Register the new rule
        recognizer = PatternRecognizer(
            supported_entity=entity_name,
            patterns=pattern_objects
        )
        
        self.analyzer.registry.add_recognizer(recognizer)
        
        # Make sure we actually search for this new thing next time
        if entity_name not in self.default_entities:
            self.default_entities.append(entity_name)
    
    def detect_pii(
        self, 
        text: str,
        entities: Optional[List[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Just Find the secrets, don't hide them.
        Useful for debugging to see what the system *would* catch.
        """
        if entities is None:
            entities = self.default_entities
        
        results = self.analyzer.analyze(
            text=text,
            entities=entities,
            language=self.language
        )
        
        # Clean up the output so it's easier to read
        return [
            {
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end]
            }
            for result in results
        ]


# A quick helper to use anywhere
def sanitize_text(text: str) -> str:
    """
    One-liner to clean text.
    Usage: clean = sanitize_text("My secret")
    """
    guard = PrivacyGuard()
    return guard.sanitize(text)
