import './App.css';
import Upload from './Upload';
import SocialProfile from './SocialProfile';
import { Card, CardHeader, CardBody, CardFooter } from '@chakra-ui/react'
function App() {
  return (
    <div className="App">
      {/* <header className="App-header">  
           
      </header> */}
      <SocialProfile></SocialProfile>
      <Card>
        <CardBody>
          <Upload url='http://localhost:8080/predict/similar_images' title="Similarity Search"></Upload>
        </CardBody>
      </Card> 
    </div>
  );
}

export default App;
