class Calculator:
    def parse_expression(self, expr):
        # list to store our tokens
        math_parts = []
        
        # go through each part of expression
        for piece in expr.split():
            # check if we got a number (normal or negative)
            if piece[0].isdigit() or (piece[0] == '-' and len(piece) > 1):
                # add it as number token
                math_parts.append({'val': float(piece), 'operation': ' ', 'number_flag': True})
            else:
                # add it as operation token
                math_parts.append({'val': 0, 'operation': piece[0], 'number_flag': False})
        return math_parts

    def evaluate_expression(self, math_parts):
        # store results after first round calculation
        working_result = []
        # keep track where we are
        pos = 0
        
        # first handle multiply, divide, modulo
        while pos < len(math_parts):
            # check for multiply/divide/modulo operations
            if not math_parts[pos]['number_flag'] and math_parts[pos]['operation'] in ['*', '/', '%']:
                # get numbers for calculation
                left_num = working_result[-1]['val']
                right_num = math_parts[pos + 1]['val']
                working_result.pop()
                
                # do math based on operation type
                if math_parts[pos]['operation'] == '*':
                    calculated = left_num * right_num
                elif math_parts[pos]['operation'] == '/':
                    calculated = left_num / right_num
                else:  # modulo here
                    calculated = left_num % right_num
                    
                # store the result
                working_result.append({'val': calculated, 'operation': ' ', 'number_flag': True})
                pos = pos + 2
            else:
                # keep other operations for later
                working_result.append(math_parts[pos])
                pos = pos + 1

        # now handle addition and subtraction
        final_value = working_result[0]['val']
        # go through remaining operations
        for i in range(1, len(working_result), 2):
            if working_result[i]['operation'] == '+':
                final_value = final_value + working_result[i + 1]['val']
            else:  # subtraction here
                final_value = final_value - working_result[i + 1]['val']

        return final_value
