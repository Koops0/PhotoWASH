import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  // This chooses a photo from the local file system and displays it
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const imageUrl = URL.createObjectURL(file)
      
      // Clean up previous object URL if it exists
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage)
      }
      
      setSelectedImage(imageUrl)
    }
  }

  // Export the image
  const handleExportImage = () => {
    if (selectedImage) {
      // Create a temporary anchor element
      const downloadLink = document.createElement('a');
      
      // Use the existing blob URL
      downloadLink.href = selectedImage;
      
      // Set a default filename
      downloadLink.download = 'exported-image.jpg';
      
      // Append to the document, trigger click, and remove
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
    }
  };
  

  useEffect(() => {
    return () => {
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage)
      }
    }
  }, [])


  return (
    <>
      <h1>PhotoWASH</h1>

      {/* File chooser */}
      <div className="file-input">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          id="image-input"
        />
      </div>
      
      {/* Image display */}
      {selectedImage && (
        <div className="image-container">
          <img 
            src={selectedImage} 
            alt="Selected" 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '600px', 
              marginTop: '20px' 
            }} 
          />
        </div>
      )}

      {/*Save Image*/}
      <div className="save-button">
        <button 
          onClick={handleExportImage}
        >
          Save Image
        </button>
      </div>
    </>
  )
}

export default App
