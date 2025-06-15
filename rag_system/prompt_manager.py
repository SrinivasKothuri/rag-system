import os
import yaml
from typing import Dict, Optional

class PromptManager:
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            # Always use the prompts directory inside the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = os.path.join(package_dir, "prompts")
        self.prompts_dir = prompts_dir
        self.templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self):
        """Load all YAML prompt templates from the prompts directory."""
        if not os.path.exists(self.prompts_dir):
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")

        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.yaml'):
                with open(os.path.join(self.prompts_dir, filename), 'r') as f:
                    template_data = yaml.safe_load(f)
                    self.templates[template_data['name']] = template_data['template']

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a prompt template by name."""
        return self.templates.get(template_name)

    def format_prompt(self, template_name: str, context: str, query: str) -> str:
        """Format a prompt using the specified template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        return template.format(context=context, query=query)

    def list_templates(self) -> Dict[str, str]:
        """List all available templates and their descriptions."""
        templates_info = {}
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.yaml'):
                with open(os.path.join(self.prompts_dir, filename), 'r') as f:
                    template_data = yaml.safe_load(f)
                    templates_info[template_data['name']] = template_data['description']
        return templates_info 