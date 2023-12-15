// src/App.js
import React, { useState } from 'react';
import FileUpload from './Components/FileUploader';
import ResultDisplay from './Components/ResultDisplay';

function App() {
	const [result, setResult] = useState(null);

	const handleFileUpload = (newResult) => {
    alert("File Uploaded !")
		setResult(newResult);
	};

	return (
		<div className='App border border-2-white h-screen w-screen flex justify-center p-12 bg-'>
			<div className='a'>
				<h1>Data Mining App</h1>
				<FileUpload onFileUpload={handleFileUpload} />
				{result && <ResultDisplay result={result} />}
			</div>
		</div>
	);
}

export default App;
