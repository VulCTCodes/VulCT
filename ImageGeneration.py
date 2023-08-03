from tree_sitter import Language, Parser, Node
from anytree import AnyNode
import glob
import argparse
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from PIL import Image


C_LANGUAGE = Language('/home/data4T2/swq/VulCT/parse/my-languages.so', 'c')
# JAVA_LANGUAGE = Language('/home/data4T2/swq/Main/parse/my-languages.so', 'java')
# PY_LANGUAGE = Language('/home/data4T2/swq/Main/parse/my-languages.so', 'python')
parser = Parser()
parser.set_language(C_LANGUAGE)



# edges={'Nexttoken':2,'Prevtoken':3,'Nextuse':4,'Prevuse':5,'If':6,'Ifelse':7,'While':8,'For':9,'Nextstmt':10,'Prevstmt':11,'Prevsib':12}
nodetype_dict = {'sizeof_expression': 0, 'preproc_arg': 1, '#if': 2, 'init_declarator': 3, 'keyword': 4, 'escape_sequence': 5, 
                 'parenthesized_expression': 6, 'translation_unit': 7, 'type_qualifier': 8, 'unary_expression': 9, 'conditional_expression': 10, 
                 '#define': 11, 'null': 12, 'return_statement': 13, 'abstract_function_declarator': 14, 'preproc_params': 15, 
                 'field_declaration_list': 16, '#ifdef': 17, 'abstract_parenthesized_declarator': 18, 'macro_type_specifier': 19, 'type_definition': 20, 
                 'field_declaration': 21, 'binary_expression': 22, 'abstract_pointer_declarator': 23, 'preproc_function_def': 24, 'variadic_parameter': 25, 
                 'parameter_declaration': 26, 'compound_literal_expression': 27, 'union_specifier': 28, 'seperator': 29, 'field_designator': 30, 
                 'labeled_statement': 31, 'string_literal': 32, 'type_descriptor': 33, 'preproc_defined': 34, 'struct_specifier': 35, 
                 'statement_identifier': 36, 'subscript_expression': 37, 'enumerator': 38, 'sized_type_specifier': 39, 'primitive_type': 40, 
                 'comma_expression': 41, 'call_expression': 42, '#endif': 43, 'preproc_elif': 44, 'cast_expression': 45, 
                 'update_expression': 46, '#else': 47, 'false': 48, 'enumerator_list': 49, 'preproc_call': 50, 
                 'parameter_list': 51, 'while_statement': 52, 'initializer_pair': 53, 'preproc_ifdef': 54, 'abstract_array_declarator': 55, 
                 '#ifndef': 56, 'declaration': 57, 'char_literal': 58, 'preproc_def': 59, 'case_statement': 60, 
                 'enum_specifier': 61, 'function_declarator': 62, 'subscript_designator': 63, 'comment': 64, 'function_definition': 65, 
                 'pointer_expression': 66, 'argument_list': 67, 'preproc_else': 68, 'compound_statement': 69, 'if_statement': 70, 
                 'identifier': 71, 'expression_statement': 72, 'field_expression': 73, 'for_statement': 74, 'do_statement': 75, 
                 'ERROR': 76, 'number_literal': 77, 'array_declarator': 78, 'attribute_specifier': 79, 'preproc_if': 80, 
                 'storage_class_specifier': 81, 'pointer_declarator': 82, 'assignment_expression': 83, 'operator': 84, 'concatenated_string': 85, 
                 'switch_statement': 86, 'break_statement': 87, 'goto_statement': 88, 'field_identifier': 89, 'initializer_list': 90, 
                 'parenthesized_declarator': 91, 'continue_statement': 92, 'preproc_directive': 93, 'type_identifier': 94, 'true': 95,
                 'bool': 96, '#elif': 97}


type_dict = {'(': 'seperator', ')': 'seperator', '{': 'seperator', '}': 'seperator', '\'': 'seperator', '\"': 'seperator',
             '[': 'seperator', ']': 'seperator', ',': 'seperator', '.': 'seperator', ';': 'seperator', ':': 'seperator',
             '+': 'operator', '-': 'operator', '*': 'operator', '/': 'operator', '++': 'operator', '--': 'operator',
             '%=': 'operator', '+=': 'operator', '->': 'operator', '?': 'operator', '<<': 'operator', '%': 'operator',
             '/=': 'operator', '<<=': 'operator', '^=': 'operator', '~': 'operator', '-=': 'operator', '|=': 'operator',
             '>>': 'operator', '>>=': 'operator', '!': 'operator', '*=': 'operator', '&=': 'operator', '|': 'operator', '^': 'operator',
             '=': 'operator', '<': 'operator', '>': 'operator', 
             '||': 'operator', '&&': 'operator', '&': 'operator', '\n': 'operator',
             '>=': 'operator', '<=': 'operator', '==': 'operator', '!=': 'operator',
             'case': 'keyword', 'switch': 'keyword', 'if': 'keyword', 'else': 'keyword',
             'do': 'keyword', 'while': 'keyword', 'for': 'keyword', 'break': 'keyword', 'continue': 'keyword',
             'default': 'keyword', 'goto': 'keyword', 'sizeof': 'keyword', 'static': 'keyword', 'return': 'keyword',
             'short': 'primitive_type', 'long': 'primitive_type', 'struct': 'primitive_type', 'unsigned': 'primitive_type',
             '...': 'parameter_declaration',
             'true': 'bool', 'false': 'bool',
             '': 'string_literal',
             }

def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    parser.add_argument('-o', '--output', help='The dir path of output matrix', type=str, required=True)
    parser.add_argument('-t', '--type', help='The type of ast : (faast,ast)', type=str, required=True)
    args = parser.parse_args()
    return args


def get_token(node):
    value = node.text.decode('utf-8')
    if value != node.type and value != '' and node.type != '\n':
        if node.type in nodetype_dict:
            token = node.type
        else:
            print(value, node.type, 'error!!!')
    elif value in type_dict:
        token = type_dict[value]
    elif value in nodetype_dict:
        token = value
    else:
        token = 'string_literal'
    return value, token


def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    # print(id)
    text, token = get_token(node)
    if id == 0:
        root.text = text
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, text=text, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in node.children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)

def get_ast(filepath):
    programfile = open(filepath, encoding='utf-8')
    programtext = bytes(programfile.read(), "utf8")

    tree = parser.parse(programtext)
    root_node = tree.root_node
    nodelist = []
    newtree = AnyNode(id=0, token=None, data=None)
    createtree(newtree, root_node, nodelist) 
    return newtree


def get_ast_edges(node,edge_list):
    for child in node.children:
        try:
            src = nodetype_dict[node.token]
        except:
            print('src',node.token, node.text)
        try:
            tgt = nodetype_dict[child.token]
        except:
            print('tgt',child.token, repr(child.text))
        edge = (src,tgt)
        # print(node.token, src, child.token, tgt)
        edge_list.append(edge)

        get_ast_edges(child,edge_list)


def get_flow_edges(node,edge_list):
    token=node.token
    # while do
    if token=='while_statement' or token=='do_statement':
        block = nodetype_dict['compound_statement']
        cond = nodetype_dict['parenthesized_expression']
        edge_list.append((block, cond))
        edge_list.append((cond, block))
    # for
    if token=='for_statement':
        init = nodetype_dict['assignment_expression']
        cond = nodetype_dict['binary_expression']
        update = nodetype_dict['update_expression']
        block = nodetype_dict['compound_statement']
        edge_list.append((init, cond))
        edge_list.append((cond, block))
        edge_list.append((block, update))
        edge_list.append((update, cond))   
    # if-else
    if token=='if_statement':
        cond = nodetype_dict['parenthesized_expression']
        cond_true = nodetype_dict['expression_statement']
        edge_list.append((cond, cond_true))
        if len(node.children)==5:
            cond_false = nodetype_dict['compound_statement']
            edge_list.append((cond, cond_false))
    # switch-case
    if token=='switch_statement':
        cond = nodetype_dict['parenthesized_expression']
        case = nodetype_dict['case_statement']
        edge_list.append((cond, case))

    for child in node.children:
        get_flow_edges(child,edge_list)


def get_next_edges(node,edge_list):
# nextsib, nextstmt
    token=node.token
    if len(node.children)>0:
        for i in range(len(node.children)-1):
            src = nodetype_dict[node.children[i].token]
            tgt = nodetype_dict[node.children[i+1].token]
            edge = (src,tgt)
            edge_list.append(edge)
            if token=='compound_statement':
                edge_list.append(edge)
    for child in node.children:
        get_next_edges(child,edge_list)


def get_nexttoken_edges(node,edge_list,tokenlist):
    def gettokenlist(node,tokenlist):
        if len(node.children)==0:
            tokenlist.append(node.token)
            # print(node.text)
        for child in node.children:
            gettokenlist(child,tokenlist)
    gettokenlist(node,tokenlist)
    for i in range(len(tokenlist)-1):
            src = nodetype_dict[tokenlist[i]]
            tgt = nodetype_dict[tokenlist[i+1]]
            edge = (src,tgt)
            edge_list.append(edge)
            

def get_nextuse_edges(node,edge_list,variabledict):
    def getvariables(node,variabledict):
        token=node.token
        if token=='identifier':
            variable = node.text
            # for child in node.children:
            #     if child.token==node.data.member:
            #         variable=child.token
            #         variablenode=child
            if not variabledict.__contains__(variable):
                variabledict[variable]=[node.id]
            else:
                variabledict[variable].append(node.id)
        for child in node.children:
            getvariables(child,variabledict)
    getvariables(node,variabledict)
    #print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
            edge_list.append((nodetype_dict['identifier'],nodetype_dict['identifier']))

def get_edges(ast, ast_type):
    edge_list = []
    get_ast_edges(ast, edge_list)
    if ast_type == 'faast':
        get_flow_edges(ast, edge_list)
        get_next_edges(ast, edge_list)
        tokenlist = []
        get_nexttoken_edges(ast, edge_list, tokenlist)
        variabledict = {}
        get_nextuse_edges(ast, edge_list, variabledict)
    elif ast_type == 'ast':
        pass
    else:
        print('type_error!!!!!!!')
    # print(edge_list)
    return edge_list

def get_matrix(file, out, ast_type):
    filename = file.split('/')[-1].split('.c')[0]
    ast = get_ast(file)
    edges = get_edges(ast, ast_type)

    NodeType_Num = len(nodetype_dict)
    matrix = [[0 for col in range(NodeType_Num)] for row in range(NodeType_Num)]
    total = len(edges)
    for src,tgt in edges:
        matrix[src][tgt] += 1
    
    for i in range(NodeType_Num):
        total = 0
        for j in range(NodeType_Num):
            total += matrix[i][j]
        if total != 0:
            for j in range(NodeType_Num):
                matrix[i][j] = matrix[i][j]/total

    matrix = np.array(matrix)
    
    # Matrix
    # outpath = out + filename +'.npy'
    # np.save(outpath, matrix)

    # Grey Image
    outpath = out + filename +'.jpg'
    img = (matrix * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.convert("L")
    img.save(outpath)


def main():
    args = parse_options()
    if args.input[-1] == '/':
        in_path = args.input
    else:
        in_path = args.input + '/'
    out_path = args.output
    folder = os.path.exists(out_path)
    if not folder:
        os.makedirs(out_path)
    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    ast_type = args.type
    types = ['NoVul','Vul']
    for type in types:
        in_dir = in_path + type + '/'
        files = glob.glob(in_dir + '*.c')
        out_dir = out_path + type + '/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        pool = Pool(10)
        pool.map(partial(get_matrix, out=out_dir, ast_type=ast_type), files)


if __name__ == '__main__':
    # main()  
    file = '/home/data4T2/swq/VulCT/code/example.c'
    # ast = get_ast(file)
    # edges = get_edges(ast, 'faast')


    get_matrix('/home/data4T2/swq/VulCT/code/example.c','/home/data4T2/swq/VulCT/code/', 'faast')



   

# input_dir = '../data/sard/*/*.c'
# files = glob.glob(input_dir)
# for file in files:
#     programfile = open(file, encoding='utf-8')
#     programtext = bytes(programfile.read(), "utf8")

#     tree = parser.parse(programtext)
#     root_node = tree.root_node
#     nodelist = []
#     newtree = AnyNode(id=0, token=None, data=None)
#     createtree(newtree, root_node, nodelist)
    

# path = '/home/data4T2/swq/Main/code/example.c'
# ast = get_ast(path)
# edges = get_faast_edges(ast)
# matrix = get_matrix(edges)
# # print(edges)







