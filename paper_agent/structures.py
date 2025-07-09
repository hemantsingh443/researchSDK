from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Author(BaseModel):
    """Represents a single author of a paper."""
    name: str
    affiliation: Optional[str] = None

class Paper(BaseModel):
    """Core data model for a scientific paper."""
    paper_id: str = Field(..., description="A unique identifier, e.g., local path, DOI, or arXiv ID")
    title: Optional[str] = None
    authors: List[Author] = []
    abstract: Optional[str] = None
    full_text: str
    
    #  will populate these in later phases
    sections: Dict[str, str] = Field(default_factory=dict, description="Text broken down by section (e.g., Introduction)")
    tables: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)

    def __repr__(self):
        return f"Paper(paper_id='{self.paper_id}', title='{self.title or 'N/A'}', authors={len(self.authors)})"

    def __str__(self):
        return self.__repr__()