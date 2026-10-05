### Correct
- Question: What position does the player who played for butler cc (ks) play?
- Gold: `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`
- Predicted: `SELECT Position FROM table WHERE School/Club Team = 'butler cc (ks)'`

### Correct
- Question: How many schools did player number 3 play at?
- Gold: `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`
- Predicted: `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`

### Correct
- Question: What school did player number 21 play for?
- Gold: `SELECT School/Club Team FROM table WHERE No. = '21'`
- Predicted: `SELECT School/Club Team FROM table WHERE No. = '21'`

### Correct
- Question: Who is the player that wears number 42?
- Gold: `SELECT Player FROM table WHERE No. = '42'`
- Predicted: `SELECT Player FROM table WHERE No. = '42'`

### Correct
- Question: Who are all of the players on the Westchester High School club team?
- Gold: `SELECT Player FROM table WHERE School/Club Team = 'Westchester High School'`
- Predicted: `SELECT Player FROM table WHERE School/Club Team = 'westchester high school'`

### Wrong: wrong condition column or operator
- Question: What player played guard for toronto in 1996-97?
- Gold: `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`
- Predicted: `SELECT Player FROM table WHERE Nationality = 'guard' AND Years in Toronto = '1996-97'`

### Wrong: wrong column
- Question: What are the total amount of numbers on the Toronto team in 2005-06?
- Gold: `SELECT COUNT(No.) FROM table WHERE Years in Toronto = '2005-06'`
- Predicted: `SELECT COUNT(Years in Toronto) FROM table WHERE School/Club Team = '2005-06'`

### Wrong: wrong column
- Question: When did Jacques Chirac stop being a G8 leader?
- Gold: `SELECT Ended time as senior G8 leader FROM table WHERE Person = 'Jacques Chirac'`
- Predicted: `SELECT Began time as senior G8 leader FROM table WHERE Entered office as Head of State or Government = 'jacques chirac stop stop'`

### Wrong: wrong column
- Question: When did the Prime Minister of Italy take office?
- Gold: `SELECT Entered office as Head of State or Government FROM table WHERE Office = 'Prime Minister of Italy'`
- Predicted: `SELECT Office FROM table WHERE Began time as senior G8 leader = 'italy'`

### Wrong: wrong condition column or operator
- Question: What official or native languages are spoken in the country whose capital city is Canberra?
- Gold: `SELECT Official or native language(s) (alphabet/script) FROM table WHERE Capital ( exonym ) = 'Canberra'`
- Predicted: `SELECT Official or native language(s) (alphabet/script) FROM table WHERE Capital ( endonym ) = 'canberra'`
