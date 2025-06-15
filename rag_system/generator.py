from typing import List, Dict, Any, Optional
from .prompt_manager import PromptManager
from .model_wrapper import create_model_wrapper
from .config import get_config

class Generator:
    def __init__(self):
        self.config = get_config()
        self.model = create_model_wrapper(self.config)
        self.prompt_manager = PromptManager()
        
    def generate(self, query: str, context: List[Dict[str, Any]], template_name: str = None) -> str:
        """
        Generate a response using the specified template.
        
        Args:
            query: The user's query
            context: List of retrieved documents with their metadata
            template_name: Name of the template to use (defaults to config value)
            
        Returns:
            Generated response
        """
        if template_name is None:
            template_name = self.config["default_template"]
            
        # Format context into a string
        context_str = self._format_context(context)
        
        # Get and format the prompt using the template
        prompt = self.prompt_manager.format_prompt(template_name, context_str, query)
        
        # Generate response using the model
        return self.model.generate_completion(prompt)
        
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format the context into a string representation."""
        formatted_context = []
        for doc in context:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Add metadata as a header if present
            if metadata:
                header = f"[Metadata: {', '.join(f'{k}={v}' for k, v in metadata.items())}]"
                formatted_context.append(f"{header}\n{content}")
            else:
                formatted_context.append(content)
                
        return "\n\n".join(formatted_context)