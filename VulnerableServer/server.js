const express = require('express')

const app = express()
const mysql = require('mysql')

app.use(express.json())

const conn = mysql.createConnection({
	host: "localhost",
	user: "root",
	password: "",
	database: "testdb"
});

conn.connect((err) => {
	if (err) throw err;
	console.log("Connected to SQL")
})

app.get('/api/v0/sqli/select', (req, res) => {
	console.log(req.query)
	const sql = req.query.user_id;
	conn.query(sql, (err, result) => {
		res.send(result);
	});
})

const server = app.listen(8081, () => {
	const port = server.address().port
	console.log(`Server is Listening at Port: ${port}`)
})

// for(let i= 0; i < 100; i++){
// 	conn.query(`delete from datatable where user_id = '${i}'`,(err,res)=>{
// 		if(err){
// 			console.log("Error at "+i);
// 			throw err;
// 		}
// 	});
// }