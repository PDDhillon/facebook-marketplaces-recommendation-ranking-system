import './App.css';
import Upload from './Upload';
import SocialProfile from './SocialProfile';
import {
  Stack,
} from '@chakra-ui/react'

const breakpoints = {
  base: "0em", // 0px
  sm: "30em", // ~480px. em is a relative unit and is dependant on the font size.
  md: "48em", // ~768px
  lg: "62em", // ~992px
  xl: "80em", // ~1280px
  "2xl": "96em", // ~1536px
};

  
function App() {
  return (
    <div className="App">
      <header className="App-header">  
        <Stack spacing={8} direction='row' m={5} width={breakpoints} wrap={"wrap"}>
            <SocialProfile></SocialProfile>
            <Upload url='http://54.170.80.153:8080/predict/similar_images' title="Similarity Search"></Upload>
        </Stack> 
      </header>
    </div>
  );
}

export default App;
