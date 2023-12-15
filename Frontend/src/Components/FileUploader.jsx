// src/components/FileUpload.js
import React, { useState } from 'react';

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
        .then(response => response.json())
        .then(result => {
            onFileUpload(result);
        })
        .catch(error => console.error('Error uploading file:', error));
    }
    onFileUpload("test");
  };

  return (
    <div className='border border-2-white'>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default FileUpload;
