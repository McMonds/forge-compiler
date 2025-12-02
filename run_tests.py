import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import tests
from tests import test_lexer
from tests import test_parser
from tests import test_semantic
from tests import test_ir_gen
from tests import test_parser_structs
from tests import test_semantic_structs
from tests import test_ir_gen_structs
from tests import test_parser_enums
from tests import test_semantic_enums
from tests import test_ir_gen_enums

def run_test(test_func):
    try:
        test_func()
        print(f"[PASS] {test_func.__name__}")
        return True
    except Exception:
        print(f"[FAIL] {test_func.__name__}")
        traceback.print_exc()
        return False

def main():
    print("Running tests...")
    tests = [
        test_lexer.test_lexer_basic,
        test_lexer.test_lexer_keywords,
        test_lexer.test_lexer_unknown_char,
        test_parser.test_parser_function,
        test_parser.test_parser_expression_precedence,
        test_parser.test_parser_error_recovery,
        test_parser_structs.test_parser_struct_definition,
        test_parser_structs.test_parser_struct_instantiation,
        test_parser_structs.test_parser_field_access,
        test_parser_enums.test_parser_enum_definition,
        test_parser_enums.test_parser_enum_instantiation,
        test_semantic.test_semantic_basic_types,
        test_semantic.test_semantic_type_mismatch,
        test_semantic.test_semantic_undefined_variable,
        test_semantic.test_semantic_scope_shadowing,
        test_ir_gen.test_ir_gen_basic,
        test_ir_gen.test_ir_gen_control_flow,
        test_semantic_structs.test_semantic_struct_definition,
        test_semantic_structs.test_semantic_struct_missing_field,
        test_semantic_structs.test_semantic_struct_field_type_mismatch,
        test_semantic_structs.test_semantic_field_access,
        test_semantic_structs.test_semantic_undefined_field,
        test_ir_gen_structs.test_ir_gen_struct_basic,
        test_ir_gen_structs.test_ir_gen_struct_field_access,
        test_semantic_enums.test_semantic_enum_definition,
        test_semantic_enums.test_semantic_enum_undefined,
        test_semantic_enums.test_semantic_enum_invalid_variant,
        test_semantic_enums.test_semantic_enum_payload_mismatch,
        test_semantic_enums.test_semantic_enum_missing_payload,
        test_ir_gen_enums.test_ir_gen_enum_basic,
    ]
    
    passed = 0
    for test in tests:
        if run_test(test):
            passed += 1
            
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed != len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()
