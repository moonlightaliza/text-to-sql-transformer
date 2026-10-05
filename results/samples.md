### Correct
- Question: How many fastest laps were there for a rd that equals 10?
- Gold: `SELECT COUNT(Fastest Lap) FROM table WHERE Rd = '10'`
- Predicted: `SELECT COUNT(Fastest Lap) FROM table WHERE Rd = '10'`

### Correct
- Question: What was the score of the game on November 12?
- Gold: `SELECT Score FROM table WHERE Date = 'November 12'`
- Predicted: `SELECT Score FROM table WHERE Date = 'november 12'`

### Correct
- Question: Who had high assists when they played against San Antonio?
- Gold: `SELECT High assists FROM table WHERE Team = 'San Antonio'`
- Predicted: `SELECT High assists FROM table WHERE Team = 'san antonio'`

### Correct
- Question: Who scored the most points in game 4?
- Gold: `SELECT High points FROM table WHERE Game = '4'`
- Predicted: `SELECT High points FROM table WHERE Game = '4'`

### Correct
- Question: Which leagues is the Galena school in?
- Gold: `SELECT League FROM table WHERE School = 'Galena'`
- Predicted: `SELECT League FROM table WHERE School = 'galena'`

### Wrong: wrong condition column or operator
- Question: What position does the player who played for butler cc (ks) play?
- Gold: `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`
- Predicted: `SELECT Position FROM table WHERE Player = 'ccksler'`

### Wrong: wrong column
- Question: How many schools did player number 3 play at?
- Gold: `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`
- Predicted: `SELECT COUNT(Player) FROM table WHERE Position = '3'`

### Wrong: wrong column
- Question: What school did player number 21 play for?
- Gold: `SELECT School/Club Team FROM table WHERE No. = '21'`
- Predicted: `SELECT Player FROM table WHERE Position = '21'`

### Wrong: wrong condition column or operator
- Question: Who is the player that wears number 42?
- Gold: `SELECT Player FROM table WHERE No. = '42'`
- Predicted: `SELECT Player FROM table WHERE School/Club Team = '42'`

### Wrong: wrong value
- Question: What player played guard for toronto in 1996-97?
- Gold: `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`
- Predicted: `SELECT Player FROM table WHERE Years in Toronto = 'guard' AND Position = '1996-97'`
