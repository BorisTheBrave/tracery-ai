# SQL Query Grammar
sql_grammar = """
root ::= select_stmt
select_stmt ::= "SELECT " select_list " FROM " table_name where_clause? order_clause? limit_clause?
select_list ::= column ("," column)*
column ::= [a-zA-Z_][a-zA-Z0-9_]* | "*"
table_name ::= [a-zA-Z_][a-zA-Z0-9_]*
where_clause ::= " WHERE " condition
condition ::= column " " op " " value
op ::= "=" | ">" | "<" | ">=" | "<=" | "!=" | "LIKE"
value ::= number | string
string ::= "'" [^']* "'"
number ::= [0-9]+ ("." [0-9]+)?
order_clause ::= " ORDER BY " column (" ASC" | " DESC")?
limit_clause ::= " LIMIT " [0-9]+
"""

# Email Grammar
email_grammar = """
root ::= email
email ::= header body
header ::= "From: " sender "\n" "To: " recipient "\n" "Subject: " subject "\n\n"
sender ::= name " <" email_address ">"
recipient ::= name " <" email_address ">"
name ::= [a-zA-Z ]+ 
email_address ::= [a-zA-Z0-9._%+-]+ "@" [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,}
subject ::= [^\n]+
body ::= paragraph ("\n\n" paragraph)* "\n\n" signature
paragraph ::= sentence+
sentence ::= [A-Z] [^.!?]* [.!?] " "
signature ::= "Best regards,\n" name
""" 