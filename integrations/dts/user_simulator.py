"""Simulate diverse user reactions to conversation strategies.

Models different user personas and intents for robust conversation testing.
"""

import random
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""
    COOPERATIVE = "cooperative"
    SKEPTICAL = "skeptical"
    CONFUSED = "confused"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    RUSHED = "rushed"
    DETAILED = "detailed"


@dataclass
class UserPersona:
    """User type definition with traits and characteristics.
    
    Attributes:
        name: Persona identifier
        traits: Dict of trait names to scores (0-1)
        knowledge_level: 'novice', 'intermediate', 'expert'
        goal_alignment: How aligned user is with conversation goal (0-1)
        communication_style: Preferred communication style
        intent_distribution: Probability distribution over IntentTypes
    """
    name: str
    traits: Dict[str, float] = field(default_factory=dict)
    knowledge_level: str = "intermediate"  # novice, intermediate, expert
    goal_alignment: float = 0.5
    communication_style: str = "direct"  # direct, verbose, terse, formal
    intent_distribution: Dict[IntentType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default traits and intent distribution."""
        if not self.traits:
            self.traits = {
                "cooperativeness": 0.5,
                "skepticism": 0.5,
                "patience": 0.5,
                "technical_affinity": 0.5,
                "formality": 0.5,
                "verbosity": 0.5,
            }
        
        if not self.intent_distribution:
            # Default distribution based on traits
            self.intent_distribution = {
                IntentType.COOPERATIVE: self.traits.get("cooperativeness", 0.5),
                IntentType.SKEPTICAL: self.traits.get("skepticism", 0.3),
                IntentType.CONFUSED: 0.2 if self.knowledge_level == "novice" else 0.1,
                IntentType.HOSTILE: 0.1 if self.goal_alignment < 0.3 else 0.0,
                IntentType.NEUTRAL: 0.3,
                IntentType.CURIOUS: 0.4 if self.knowledge_level != "expert" else 0.2,
                IntentType.RUSHED: 0.15 if self.traits.get("patience", 0.5) < 0.4 else 0.05,
                IntentType.DETAILED: 0.3 if self.traits.get("verbosity", 0.5) > 0.6 else 0.1,
            }
    
    def get_dominant_intent(self) -> IntentType:
        """Get the most likely intent for this persona."""
        return max(self.intent_distribution, key=self.intent_distribution.get)
    
    def sample_intent(self) -> IntentType:
        """Sample an intent based on distribution."""
        intents = list(self.intent_distribution.keys())
        weights = list(self.intent_distribution.values())
        return random.choices(intents, weights=weights)[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            "name": self.name,
            "traits": self.traits,
            "knowledge_level": self.knowledge_level,
            "goal_alignment": self.goal_alignment,
            "communication_style": self.communication_style,
            "dominant_intent": self.get_dominant_intent().value,
        }


# Predefined personas for common scenarios
PREDEFINED_PERSONAS = {
    "cooperative_user": UserPersona(
        name="Cooperative User",
        traits={"cooperativeness": 0.9, "skepticism": 0.2, "patience": 0.8},
        goal_alignment=0.9,
        knowledge_level="intermediate",
    ),
    "skeptical_expert": UserPersona(
        name="Skeptical Expert",
        traits={"cooperativeness": 0.3, "skepticism": 0.9, "patience": 0.6, "technical_affinity": 0.95},
        goal_alignment=0.4,
        knowledge_level="expert",
        communication_style="terse",
    ),
    "confused_beginner": UserPersona(
        name="Confused Beginner",
        traits={"cooperativeness": 0.7, "skepticism": 0.1, "patience": 0.3, "technical_affinity": 0.2},
        goal_alignment=0.6,
        knowledge_level="novice",
        communication_style="verbose",
    ),
    "hostile_critic": UserPersona(
        name="Hostile Critic",
        traits={"cooperativeness": 0.1, "skepticism": 0.9, "patience": 0.2, "formality": 0.3},
        goal_alignment=0.1,
        knowledge_level="intermediate",
    ),
    "curious_explorer": UserPersona(
        name="Curious Explorer",
        traits={"cooperativeness": 0.8, "skepticism": 0.4, "patience": 0.9, "verbosity": 0.8},
        goal_alignment=0.7,
        knowledge_level="intermediate",
    ),
    "time_constrained": UserPersona(
        name="Time-Constrained User",
        traits={"cooperativeness": 0.6, "patience": 0.1, "verbosity": 0.2},
        goal_alignment=0.5,
        knowledge_level="intermediate",
        communication_style="terse",
    ),
    "formal_professional": UserPersona(
        name="Formal Professional",
        traits={"cooperativeness": 0.7, "skepticism": 0.5, "formality": 0.95, "patience": 0.7},
        goal_alignment=0.6,
        knowledge_level="expert",
        communication_style="formal",
    ),
    "enthusiastic_early_adopter": UserPersona(
        name="Enthusiastic Early Adopter",
        traits={"cooperativeness": 0.95, "skepticism": 0.1, "technical_affinity": 0.8, "patience": 0.7},
        goal_alignment=0.9,
        knowledge_level="intermediate",
    ),
}


@dataclass
class IntentModel:
    """Intent classification and satisfaction detection.
    
    Analyzes conversation history to understand user intent
    and detect satisfaction levels.
    """
    
    def classify_intent(self, message: str) -> Tuple[IntentType, float]:
        """Classify the intent of a user message.
        
        Args:
            message: User message to classify
            
        Returns:
            Tuple of (IntentType, confidence)
        """
        message_lower = message.lower()
        
        # Simple keyword-based classification
        scores = {
            IntentType.COOPERATIVE: 0.0,
            IntentType.SKEPTICAL: 0.0,
            IntentType.CONFUSED: 0.0,
            IntentType.HOSTILE: 0.0,
            IntentType.NEUTRAL: 0.3,  # Base score
            IntentType.CURIOUS: 0.0,
            IntentType.RUSHED: 0.0,
            IntentType.DETAILED: 0.0,
        }
        
        # Cooperative indicators
        coop_words = ["yes", "sure", "okay", "great", "thanks", "agree", "helpful"]
        scores[IntentType.COOPERATIVE] += sum(1 for w in coop_words if w in message_lower) * 0.3
        
        # Skeptical indicators
        skept_words = ["but", "however", "why", "doubt", "uncertain", "really", "actually"]
        scores[IntentType.SKEPTICAL] += sum(1 for w in skept_words if w in message_lower) * 0.3
        
        # Confused indicators
        confused_words = ["confused", "don't understand", "unclear", "what do you mean", "?"]
        scores[IntentType.CONFUSED] += sum(1 for w in confused_words if w in message_lower) * 0.25
        scores[IntentType.CONFUSED] += message.count("?") * 0.1
        
        # Hostile indicators
        hostile_words = ["wrong", "bad", "terrible", "hate", "awful", "useless", "stupid"]
        scores[IntentType.HOSTILE] += sum(1 for w in hostile_words if w in message_lower) * 0.4
        
        # Curious indicators
        curious_words = ["how", "what if", "explain", "tell me more", "interested", "curious"]
        scores[IntentType.CURIOUS] += sum(1 for w in curious_words if w in message_lower) * 0.3
        
        # Rushed indicators
        rushed_words = ["quick", "fast", "hurry", "short", "brief", "tl;dr", "summary"]
        scores[IntentType.RUSHED] += sum(1 for w in rushed_words if w in message_lower) * 0.35
        
        # Detailed indicators
        detailed_words = ["details", "elaborate", "specific", "in depth", "thorough", "comprehensive"]
        scores[IntentType.DETAILED] += sum(1 for w in detailed_words if w in message_lower) * 0.35
        
        # Get highest scoring intent
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent], 1.0)
        
        return best_intent, confidence
    
    def detect_satisfaction(self, history: List[Dict[str, str]]) -> float:
        """Detect user satisfaction level from conversation history.
        
        Args:
            history: List of conversation turns
            
        Returns:
            Satisfaction score (0-1)
        """
        if not history:
            return 0.5
        
        satisfaction = 0.5  # Neutral starting point
        
        # Positive indicators
        positive_words = [
            "thank", "good", "great", "excellent", "perfect", "helpful", 
            "clear", "understand", "yes", "agree", "satisfied", "happy"
        ]
        
        # Negative indicators
        negative_words = [
            "bad", "wrong", "terrible", "confused", "unclear", "frustrated",
            "annoying", "useless", "no", "disagree", "unsatisfied", "unhappy"
        ]
        
        # Analyze recent messages (last 3 turns)
        recent = history[-3:]
        
        for turn in recent:
            if turn.get("speaker") == "user":
                msg = turn.get("message", "").lower()
                
                positive_count = sum(1 for w in positive_words if w in msg)
                negative_count = sum(1 for w in negative_words if w in msg)
                
                satisfaction += (positive_count * 0.1) - (negative_count * 0.15)
        
        # Trend analysis
        if len(history) >= 3:
            early_intent, _ = self.classify_intent(history[0].get("message", ""))
            recent_intent, _ = self.classify_intent(history[-1].get("message", ""))
            
            # Positive transition
            if early_intent in [IntentType.CONFUSED, IntentType.SKEPTICAL] and \
               recent_intent in [IntentType.COOPERATIVE, IntentType.CURIOUS]:
                satisfaction += 0.1
            
            # Negative transition
            if early_intent in [IntentType.COOPERATIVE] and \
               recent_intent in [IntentType.HOSTILE, IntentType.SKEPTICAL]:
                satisfaction -= 0.1
        
        return max(0.0, min(1.0, satisfaction))
    
    def detect_goal_progress(self, history: List[Dict[str, str]], goal: str) -> float:
        """Detect progress toward conversation goal.
        
        Args:
            history: Conversation history
            goal: Target goal
            
        Returns:
            Progress score (0-1)
        """
        if not history:
            return 0.0
        
        # Simple heuristic: look for goal-related keywords
        goal_keywords = set(goal.lower().split())
        
        agreement_signals = [
            "agree", "accept", "yes", "sure", "okay", "will do", "makes sense",
            "understand", "clear", "got it"
        ]
        
        progress = 0.0
        
        # Count goal mentions and agreement signals
        for turn in history:
            msg = turn.get("message", "").lower()
            
            # Goal relevance
            goal_mentions = sum(1 for kw in goal_keywords if kw in msg)
            progress += goal_mentions * 0.05
            
            # Agreement signals
            if turn.get("speaker") == "user":
                agreements = sum(1 for sig in agreement_signals if sig in msg)
                progress += agreements * 0.1
        
        return min(1.0, progress)


@dataclass
class UserSimulator:
    """Simulate user responses to conversation strategies.
    
    Uses personas to generate diverse user reactions for testing.
    """
    personas: List[UserPersona] = field(default_factory=list)
    intent_model: IntentModel = field(default_factory=IntentModel)
    llm_client: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize default personas if none provided."""
        if not self.personas:
            self.personas = list(PREDEFINED_PERSONAS.values())[:4]  # Use first 4
    
    def add_persona(self, persona: UserPersona) -> None:
        """Add a persona to the simulator."""
        self.personas.append(persona)
    
    def get_persona(self, name: str) -> Optional[UserPersona]:
        """Get a persona by name."""
        for persona in self.personas:
            if persona.name == name:
                return persona
        return None
    
    def simulate_response(
        self, 
        strategy: str, 
        persona: Optional[UserPersona] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Simulate a user response to a conversation strategy.
        
        Args:
            strategy: The conversation strategy/messages
            persona: User persona to simulate (random if None)
            context: Additional context for simulation
            conversation_history: Previous conversation turns
            
        Returns:
            Simulated user response
        """
        if persona is None:
            persona = random.choice(self.personas)
        
        history = conversation_history or []
        
        # Determine intent based on persona and current state
        intent = persona.sample_intent()
        
        # Generate response based on intent and persona traits
        response = self._generate_response_for_intent(
            intent, strategy, persona, history, context
        )
        
        return response
    
    def _generate_response_for_intent(
        self,
        intent: IntentType,
        strategy: str,
        persona: UserPersona,
        history: List[Dict[str, str]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a response matching the intent."""
        
        # Template responses based on intent
        responses = {
            IntentType.COOPERATIVE: [
                "That sounds great! How do we proceed?",
                "Yes, that makes perfect sense. I'm on board.",
                "Excellent suggestion. What's the next step?",
                "I agree with that approach. Let's do it.",
                "That helps a lot, thank you!",
            ],
            IntentType.SKEPTICAL: [
                "I'm not entirely convinced. Can you explain further?",
                "How do you know that will work?",
                "What about the risks involved?",
                "I'm not sure that's the best approach.",
                "Can you provide more evidence for that?",
            ],
            IntentType.CONFUSED: [
                "I'm not sure I understand. Could you clarify?",
                "What do you mean by that?",
                "I'm a bit lost. Can you explain more simply?",
                "Sorry, I don't follow. Could you rephrase?",
                "That seems complicated. Can you break it down?",
            ],
            IntentType.HOSTILE: [
                "That doesn't help at all.",
                "This is a waste of time.",
                "You're not addressing my actual problem.",
                "I expected better than this.",
                "This approach is clearly flawed.",
            ],
            IntentType.NEUTRAL: [
                "I see. What else can you tell me?",
                "Understood. Go on.",
                "That's noted. What's next?",
                "I understand. Any other information?",
                "Okay. Please continue.",
            ],
            IntentType.CURIOUS: [
                "That's interesting! How does that work?",
                "Can you tell me more about that?",
                "What happens if we try a different approach?",
                "I'm curious about the details. Can you elaborate?",
                "How does this compare to other methods?",
            ],
            IntentType.RUSHED: [
                "Can you give me the short version?",
                "Just the key points, please.",
                "Quickly - what's the main takeaway?",
                "I need a brief summary.",
                "TL;DR?",
            ],
            IntentType.DETAILED: [
                "Can you be more specific about the implementation?",
                "I'd like to understand all the details.",
                "What are the exact steps involved?",
                "Please provide a thorough explanation.",
                "Can you elaborate on each component?",
            ],
        }
        
        # Select base response
        base_responses = responses.get(intent, responses[IntentType.NEUTRAL])
        response = random.choice(base_responses)
        
        # Modify based on knowledge level
        if persona.knowledge_level == "novice":
            if intent == IntentType.CONFUSED:
                response = "I'm really lost here. Could you explain that like I'm completely new to this?"
            elif intent == IntentType.CURIOUS:
                response = "I want to learn more, but please keep it simple."
        elif persona.knowledge_level == "expert":
            if intent == IntentType.SKEPTICAL:
                response += " I've seen similar approaches fail before."
            elif intent == IntentType.CURIOUS:
                response = f"I'm familiar with the basics. {response}"
        
        # Modify based on communication style
        if persona.communication_style == "terse":
            response = response.split('.')[0] + '.' if '.' in response else response[:50]
        elif persona.communication_style == "verbose":
            response += " I'd really appreciate a comprehensive explanation that covers all the angles."
        elif persona.communication_style == "formal":
            response = "I would like to request " + response.lower()
        
        return response
    
    def generate_intent_variants(self, strategy: str, k: int = 3) -> List[str]:
        """Generate K user intent variants for a strategy.
        
        Args:
            strategy: The conversation strategy to respond to
            k: Number of variants to generate
            
        Returns:
            List of simulated user responses
        """
        variants = []
        
        # Select diverse personas
        selected_personas = random.sample(
            self.personas, 
            min(k, len(self.personas))
        )
        
        for persona in selected_personas:
            response = self.simulate_response(strategy, persona)
            variants.append(response)
        
        # If we need more variants, sample additional intents
        while len(variants) < k:
            persona = random.choice(self.personas)
            response = self.simulate_response(strategy, persona)
            if response not in variants:
                variants.append(response)
        
        return variants[:k]
    
    def simulate_conversation_turns(
        self,
        strategy_sequence: List[str],
        persona: Optional[UserPersona] = None,
        max_turns: int = 5
    ) -> List[Dict[str, str]]:
        """Simulate a full multi-turn conversation.
        
        Args:
            strategy_sequence: List of system strategies/messages
            persona: User persona to simulate
            max_turns: Maximum conversation turns
            
        Returns:
            Conversation history
        """
        if persona is None:
            persona = random.choice(self.personas)
        
        history = []
        
        for i, strategy in enumerate(strategy_sequence[:max_turns]):
            # System turn
            history.append({"speaker": "system", "message": strategy})
            
            # User turn
            response = self.simulate_response(
                strategy, 
                persona, 
                conversation_history=history
            )
            history.append({"speaker": "user", "message": response})
            
            # Check if user seems satisfied enough to end early
            satisfaction = self.intent_model.detect_satisfaction(history)
            if satisfaction > 0.8 and i >= 2:
                break
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulator statistics."""
        return {
            "num_personas": len(self.personas),
            "persona_names": [p.name for p in self.personas],
            "avg_goal_alignment": sum(p.goal_alignment for p in self.personas) / len(self.personas) if self.personas else 0,
        }


# Type hint for Any
from typing import Any
