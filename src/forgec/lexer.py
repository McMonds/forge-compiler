import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Any
from forgec.diagnostics import Span, DiagnosticEngine

class TokenType(Enum):
    # Keywords
    LET = auto()
    IF = auto()
    ELSE = auto()
    FN = auto()
    RETURN = auto()
    TRUE = auto()
    FALSE = auto()
    STRUCT = auto()
    ENUM = auto()
    MATCH = auto()
    TRAIT = auto()
    IMPL = auto()
    FOR = auto()
    SELF = auto()
    MOD = auto()
    USE = auto()
    PUB = auto()
    EXTERN = auto()
    MUT = auto()

    # Literals
    INTEGER = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Operators & Punctuation
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    EQ = auto()        # =
    EQEQ = auto()      # ==
    NEQ = auto()       # !=
    LT = auto()        # <
    GT = auto()        # >
    DOT = auto()       # .
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    LBRACE = auto()    # {
    RBRACE = auto()    # }
    COLON = auto()     # :
    COLONCOLON = auto() # ::
    AMPERSAND = auto()
    SEMICOLON = auto() # ;
    COMMA = auto()     # ,
    ARROW = auto()     # ->
    FATARROW = auto()  # =>

    # Special
    EOF = auto()
    ERROR = auto()

@dataclass
class Token:
    type: TokenType
    lexeme: str
    span: Span
    value: Optional[Any] = None

class Lexer:
    def __init__(self, source: str, diagnostics: DiagnosticEngine):
        self.source = source
        self.diagnostics = diagnostics
        self.tokens: List[Token] = []
        self.current_pos = 0
        self.line = 1
        self.column = 1

        # Regex patterns
        self.patterns = [
            (TokenType.LET, r'\blet\b'),
            (TokenType.IF, r'\bif\b'),
            (TokenType.ELSE, r'\belse\b'),
            (TokenType.FN, r'\bfn\b'),
            (TokenType.RETURN, r'\breturn\b'),
            (TokenType.TRUE, r'\btrue\b'),
            (TokenType.FALSE, r'\bfalse\b'),
            (TokenType.STRUCT, r'\bstruct\b'),
            (TokenType.ENUM, r'\benum\b'),
            (TokenType.MATCH, r'\bmatch\b'),
            (TokenType.TRAIT, r'\btrait\b'),
            (TokenType.IMPL, r'\bimpl\b'),
            (TokenType.FOR, r'\bfor\b'),
            (TokenType.SELF, r'\bself\b'),
            (TokenType.MOD, r'\bmod\b'),
            (TokenType.USE, r'\buse\b'),
            (TokenType.PUB, r'\bpub\b'),
            (TokenType.EXTERN, r'\bextern\b'),
            (TokenType.MUT, r'\bmut\b'),
            
            (TokenType.FATARROW, r'=>'),
            (TokenType.ARROW, r'->'),
            (TokenType.EQEQ, r'=='),
            (TokenType.NEQ, r'!='),
            (TokenType.EQ, r'='),
            (TokenType.LT, r'<'),
            (TokenType.GT, r'>'),
            (TokenType.PLUS, r'\+'),
            (TokenType.MINUS, r'-'),
            (TokenType.STAR, r'\*'),
            (TokenType.SLASH, r'/'),
            (TokenType.LPAREN, r'\('),
            (TokenType.RPAREN, r'\)'),
            (TokenType.LBRACE, r'\{'),
            (TokenType.RBRACE, r'\}'),
            (TokenType.COLONCOLON, r'::'),
            (TokenType.AMPERSAND, r'&'),
            (TokenType.COLON, r':'),
            (TokenType.SEMICOLON, r';'),
            (TokenType.COMMA, r','),
            (TokenType.DOT, r'\.'),

            (TokenType.INTEGER, r'\d+'),
            (TokenType.STRING, r'"[^"]*"'),
            (TokenType.IDENTIFIER, r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ]
        self.skip_pattern = re.compile(r'\s+|//.*') # Skip whitespace and comments

    def tokenize(self) -> List[Token]:
        while self.current_pos < len(self.source):
            # Skip whitespace and comments
            match = self.skip_pattern.match(self.source, self.current_pos)
            if match:
                self._advance(match.end() - self.current_pos)
                continue

            if self.current_pos >= len(self.source):
                break

            matched = False
            for token_type, pattern in self.patterns:
                regex = re.compile(pattern)
                match = regex.match(self.source, self.current_pos)
                if match:
                    lexeme = match.group(0)
                    span = Span(self.current_pos, self.current_pos + len(lexeme), self.line, self.column)
                    
                    value = None
                    if token_type == TokenType.INTEGER:
                        value = int(lexeme)
                    elif token_type == TokenType.STRING:
                        value = lexeme[1:-1] # Strip quotes
                    
                    self.tokens.append(Token(token_type, lexeme, span, value))
                    self._advance(len(lexeme))
                    matched = True
                    break
            
            if not matched:
                # Error handling for unknown character
                char = self.source[self.current_pos]
                span = Span(self.current_pos, self.current_pos + 1, self.line, self.column)
                self.diagnostics.error(f"Unexpected character: '{char}'", span)
                # Emit error token to allow parsing to potentially continue or just fail gracefully
                self.tokens.append(Token(TokenType.ERROR, char, span))
                self._advance(1)

        # EOF Token
        span = Span(self.current_pos, self.current_pos, self.line, self.column)
        self.tokens.append(Token(TokenType.EOF, "", span))
        return self.tokens

    def _advance(self, amount: int):
        # Update line/col tracking
        text = self.source[self.current_pos : self.current_pos + amount]
        for char in text:
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        self.current_pos += amount
