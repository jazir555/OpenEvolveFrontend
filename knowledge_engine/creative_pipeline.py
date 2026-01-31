"""
Creative Pipeline Module for OpenEvolve Knowledge Engine

Specialized pipeline for creative writing and narrative generation.
Provides story structure templates and creative enhancement.
"""

import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class CreativeFormat(Enum):
    """Creative writing formats"""
    SHORT_STORY = "short_story"
    POEM = "poem"
    DIALOGUE = "dialogue"
    DESCRIPTION = "description"
    SCENE = "scene"
    CHARACTER_SKETCH = "character_sketch"


@dataclass
class StoryStructure:
    """Story structure template"""
    name: str
    stages: List[str]
    prompt_additions: Dict[str, str]


class CreativeEnhancer:
    """
    Enhances creative prompts with structure and guidance.
    """
    
    STORY_STRUCTURES = {
        'three_act': StoryStructure(
            name="Three-Act Structure",
            stages=["Setup", "Confrontation", "Resolution"],
            prompt_additions={
                'setup': 'Introduce the protagonist and their world. Present the inciting incident.',
                'confrontation': 'Develop rising action and complications. Build to a climax.',
                'resolution': 'Show the aftermath and character transformation.'
            }
        ),
        'heros_journey': StoryStructure(
            name="Hero's Journey",
            stages=["Ordinary World", "Call to Adventure", "Trials", "Return"],
            prompt_additions={
                'ordinary': 'Show the hero in their familiar environment.',
                'call': 'Present the challenge or opportunity that disrupts their world.',
                'trials': 'Describe the obstacles faced and allies gained.',
                'return': 'Show how the hero has changed and brings wisdom back.'
            }
        ),
        'five_act': StoryStructure(
            name="Five-Act Structure",
            stages=["Exposition", "Rising Action", "Climax", "Falling Action", "Denouement"],
            prompt_additions={
                'exposition': 'Set the scene and introduce characters.',
                'rising': 'Build tension through complications.',
                'climax': 'The turning point of maximum tension.',
                'falling': 'Consequences of the climax unfold.',
                'denouement': 'Resolution and loose ends tied.'
            }
        )
    }
    
    CREATIVE_TECHNIQUES = [
        "Show, don't tell - use sensory details",
        "Use vivid, specific imagery",
        "Include emotional beats and reactions",
        "Vary sentence length for rhythm",
        "Use dialogue to reveal character",
        "Create tension through conflict",
        "Include specific, concrete details",
        "Appeal to multiple senses"
    ]
    
    def __init__(self):
        self.default_structure = 'three_act'
        self.default_temperature = 0.8
        self.default_max_tokens = 1200
        
    def enhance(self, prompt: str, format_type: CreativeFormat = None,
               structure: str = None) -> Dict[str, Any]:
        """
        Enhance a creative prompt with structure and techniques.
        
        Args:
            prompt: Original creative prompt
            format_type: Type of creative format
            structure: Story structure to use
            
        Returns:
            Dict with enhanced prompt and parameters
        """
        # Detect format if not specified
        if not format_type:
            format_type = self._detect_format(prompt)
        
        # Select structure
        structure_key = structure or self.default_structure
        story_structure = self.STORY_STRUCTURES.get(structure_key, self.STORY_STRUCTURES['three_act'])
        
        # Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(prompt, format_type, story_structure)
        
        # Select techniques
        selected_techniques = random.sample(self.CREATIVE_TECHNIQUES, 
                                          min(4, len(self.CREATIVE_TECHNIQUES)))
        
        return {
            'original_prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'format': format_type.value,
            'structure': story_structure.name,
            'techniques': selected_techniques,
            'parameters': {
                'temperature': self.default_temperature,
                'max_tokens': self.default_max_tokens,
                'top_p': 0.95,
                'frequency_penalty': 0.3,  # Encourage variety
                'presence_penalty': 0.3
            }
        }
    
    def _detect_format(self, prompt: str) -> CreativeFormat:
        """Detect creative format from prompt"""
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ['poem', 'poetry', 'verse', 'rhyme']):
            return CreativeFormat.POEM
        
        if any(w in prompt_lower for w in ['dialogue', 'conversation', 'script']):
            return CreativeFormat.DIALOGUE
        
        if any(w in prompt_lower for w in ['character', 'person', 'profile']):
            return CreativeFormat.CHARACTER_SKETCH
        
        if any(w in prompt_lower for w in ['scene', 'setting', 'place', 'location']):
            return CreativeFormat.SCENE
        
        if any(w in prompt_lower for w in ['describe', 'description', 'portray']):
            return CreativeFormat.DESCRIPTION
        
        # Default to short story
        return CreativeFormat.SHORT_STORY
    
    def _build_enhanced_prompt(self, prompt: str, format_type: CreativeFormat,
                              structure: StoryStructure) -> str:
        """Build the enhanced creative prompt"""
        
        enhanced = f"Creative Writing Task: {format_type.value.replace('_', ' ').title()}\n\n"
        enhanced += f"Original Request: {prompt}\n\n"
        
        # Add structure guidance
        enhanced += f"Structure ({structure.name}):\n"
        for i, stage in enumerate(structure.stages, 1):
            addition = structure.prompt_additions.get(stage.lower().replace(' ', '_'), '')
            enhanced += f"  {i}. {stage}: {addition}\n"
        
        enhanced += "\nWriting Guidelines:\n"
        for technique in random.sample(self.CREATIVE_TECHNIQUES, 3):
            enhanced += f"  - {technique}\n"
        
        enhanced += "\nPlease write an original, engaging piece following this structure."
        
        return enhanced
    
    def get_format_specific_instructions(self, format_type: CreativeFormat) -> str:
        """Get format-specific writing instructions"""
        
        instructions = {
            CreativeFormat.SHORT_STORY: """
- Focus on a single compelling character arc
- Include sensory details and emotional depth
- Build to a satisfying conclusion""",
            
            CreativeFormat.POEM: """
- Consider rhythm and meter
- Use metaphor and imagery
- Create emotional resonance through word choice""",
            
            CreativeFormat.DIALOGUE: """
- Each character should have a distinct voice
- Use dialogue tags sparingly
- Show personality through speech patterns""",
            
            CreativeFormat.DESCRIPTION: """
- Engage all five senses
- Use specific, concrete details
- Create a vivid mental image""",
            
            CreativeFormat.SCENE: """
- Establish setting clearly
- Include action and reaction
- Show character through behavior""",
            
            CreativeFormat.CHARACTER_SKETCH: """
- Show personality through details
- Include physical and psychological traits
- Reveal through actions and speech"""
        }
        
        return instructions.get(format_type, instructions[CreativeFormat.SHORT_STORY])


class StoryPromptBuilder:
    """
    Builder for constructing detailed story prompts.
    """
    
    def __init__(self):
        self.genre = None
        self.tone = None
        self.characters = []
        self.setting = None
        self.conflict = None
        self.theme = None
        self.length = None
        
    def set_genre(self, genre: str) -> 'StoryPromptBuilder':
        """Set story genre"""
        self.genre = genre
        return self
    
    def set_tone(self, tone: str) -> 'StoryPromptBuilder':
        """Set story tone"""
        self.tone = tone
        return self
    
    def add_character(self, name: str, traits: List[str]) -> 'StoryPromptBuilder':
        """Add a character"""
        self.characters.append({'name': name, 'traits': traits})
        return self
    
    def set_setting(self, setting: str, details: str = None) -> 'StoryPromptBuilder':
        """Set story setting"""
        self.setting = {'location': setting, 'details': details}
        return self
    
    def set_conflict(self, conflict: str) -> 'StoryPromptBuilder':
        """Set central conflict"""
        self.conflict = conflict
        return self
    
    def set_theme(self, theme: str) -> 'StoryPromptBuilder':
        """Set story theme"""
        self.theme = theme
        return self
    
    def set_length(self, words: int) -> 'StoryPromptBuilder':
        """Set target length"""
        self.length = words
        return self
    
    def build(self) -> str:
        """Build the complete story prompt"""
        prompt_parts = []
        
        if self.genre:
            prompt_parts.append(f"Genre: {self.genre}")
        
        if self.tone:
            prompt_parts.append(f"Tone: {self.tone}")
        
        if self.setting:
            setting_str = f"Setting: {self.setting['location']}"
            if self.setting['details']:
                setting_str += f" - {self.setting['details']}"
            prompt_parts.append(setting_str)
        
        if self.characters:
            char_str = "Characters:\n"
            for char in self.characters:
                char_str += f"  - {char['name']}: {', '.join(char['traits'])}\n"
            prompt_parts.append(char_str.strip())
        
        if self.conflict:
            prompt_parts.append(f"Central Conflict: {self.conflict}")
        
        if self.theme:
            prompt_parts.append(f"Theme: {self.theme}")
        
        prompt = "Write a story with the following elements:\n\n"
        prompt += "\n".join(prompt_parts)
        
        if self.length:
            prompt += f"\n\nTarget length: approximately {self.length} words"
        
        prompt += "\n\nMake the story engaging with vivid descriptions and emotional depth."
        
        return prompt


# Convenience functions
def enhance_creative_prompt(prompt: str, format_type: str = None, 
                           structure: str = None) -> Dict[str, Any]:
    """Quick creative prompt enhancement"""
    enhancer = CreativeEnhancer()
    
    fmt = None
    if format_type:
        try:
            fmt = CreativeFormat(format_type)
        except ValueError:
            pass
    
    return enhancer.enhance(prompt, fmt, structure)


def build_story_prompt(**kwargs) -> str:
    """Quick story prompt builder"""
    builder = StoryPromptBuilder()
    
    if 'genre' in kwargs:
        builder.set_genre(kwargs['genre'])
    if 'tone' in kwargs:
        builder.set_tone(kwargs['tone'])
    if 'setting' in kwargs:
        builder.set_setting(kwargs['setting'], kwargs.get('setting_details'))
    if 'conflict' in kwargs:
        builder.set_conflict(kwargs['conflict'])
    if 'theme' in kwargs:
        builder.set_theme(kwargs['theme'])
    if 'length' in kwargs:
        builder.set_length(kwargs['length'])
    if 'characters' in kwargs:
        for char in kwargs['characters']:
            builder.add_character(char['name'], char.get('traits', []))
    
    return builder.build()


if __name__ == "__main__":
    # Test creative enhancer
    print("=" * 70)
    print("CREATIVE ENHANCER TESTS")
    print("=" * 70)
    
    enhancer = CreativeEnhancer()
    
    # Test 1: Short story
    prompt1 = "Write a story about an AI that discovers emotions"
    result1 = enhancer.enhance(prompt1)
    
    print(f"\nTest 1: {prompt1}")
    print(f"Format: {result1['format']}")
    print(f"Structure: {result1['structure']}")
    print(f"Temperature: {result1['parameters']['temperature']}")
    print(f"\nEnhanced prompt preview:\n{result1['enhanced_prompt'][:300]}...")
    
    # Test 2: Poem detection
    prompt2 = "Write a poem about the changing seasons"
    result2 = enhancer.enhance(prompt2)
    
    print(f"\n\nTest 2: {prompt2}")
    print(f"Detected format: {result2['format']}")
    
    # Test story builder
    print("\n" + "=" * 70)
    print("STORY PROMPT BUILDER TEST")
    print("=" * 70)
    
    story_prompt = build_story_prompt(
        genre="Science Fiction",
        tone="Thoughtful and melancholic",
        setting="A space station orbiting a dying star",
        conflict="The protagonist must choose between saving the station or themselves",
        theme="Sacrifice and humanity",
        length=800,
        characters=[
            {'name': 'Dr. Elena Vasquez', 'traits': ['brilliant', 'haunted by past decisions']},
            {'name': 'AI Companion K-9', 'traits': ['loyal', 'developing self-awareness']}
        ]
    )
    
    print(f"\nGenerated prompt:\n{story_prompt}")
