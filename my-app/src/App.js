import './App.css';
import Upload from './Upload';
import SocialProfile from './SocialProfile';
import {
  Stack, Box, useColorMode, Button
} from '@chakra-ui/react'

function App() {
  
  const { toggleColorMode } = useColorMode()
  return (
    <div className="App">
      <Box
        margin={{ base: "1rem", md: "3rem auto" }}
        maxWidth={{ base: "100%", md: "32rem" }}
      >
        <Button size='sm' onClick={toggleColorMode}>
        Toggle Mode
      </Button>
        <Stack
          marginTop="1rem"
          direction={{ base: "column", md: "row" }}
          gap="1rem"
          alignItems="stretch"
          justifyContent={"center"}
        >
          <SocialProfile></SocialProfile>
          <Upload url='https://54.170.80.153:8080/predict/similar_images' title="Similarity Search"></Upload>
        </Stack>
      </Box>
    </div>
  );
}

export default App;
