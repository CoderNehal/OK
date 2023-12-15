// src/components/FileUpload.js
import axios from 'axios';
import React, { useEffect, useState } from 'react';

const FileUpload = ({ onFileUpload }) => {
	const [file, setFile] = useState(null);

	const handleFileChange = (event) => {
		const selectedFile = event.target.files[0];
		setFile(selectedFile);
	};

	const handleUpload = () => {
		if (file) {
			const formData = new FormData();
			formData.append('file', file);

			fetch('http://localhost:8000/upload/', {
				method: 'POST',
				body: formData,
			})
				.then((response) => response.json())
				.then((result) => {
					onFileUpload(result);
				})
				.catch((error) => console.error('Error uploading file:', error));
		}
		// onFileUpload('test');
	};

	useEffect(() => {
		axios.get('http://localhost:8000/plot_dendrogram/').then((response1) => {
			console.log('Dendrogram Image:', response1.data);

			axios.get('http://localhost:8000/plot_kMeans/').then((response2) => {
				console.log('KMeans Image:', response2.data);

				axios.get('http://localhost:8000/plot_kMedoids/').then((response3) => {
					console.log('KMedoids Image:', response3.data);

					axios.get('http://localhost:8000/plot_BIRCH/').then((response4) => {
						console.log('BIRCH Image:', response4.data);

						axios
							.get('http://localhost:8000/plot_DBSCAN/')
							.then((response5) => {
								console.log('DBSCAN Image:', response5.data);
							});
					});
				});
			});
		});
	}, []);

	return (
		<div className='border border-2-white'>
			<input type='file' onChange={handleFileChange} />
			<button onClick={handleUpload}>Upload</button>
		</div>
	);
};

export default FileUpload;
