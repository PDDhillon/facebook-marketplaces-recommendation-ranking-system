import './App.css';
import Upload from './Upload';
import SocialProfile from './SocialProfile';
import {
  Stack,
} from '@chakra-ui/react'
function App() {
  return (
    <div className="App">
      {/* <header className="App-header">  
           
      </header> */}
      <Stack spacing={8} direction='row' m={5}>
          <SocialProfile></SocialProfile>
          <Upload url='http://localhost:8080/predict/similar_images' title="Similarity Search"></Upload>
      </Stack>
       
    </div>
  );
}

export default App;
