import './App.css';
import ImageCard from './ImageCard';
import { useState } from 'react';
import {
    Box,
    useColorModeValue,
    SimpleGrid,
    useDisclosure,
    Collapse,
    Heading,
    Center,
    Button,
    Input,
    Image,
    Card,
    CardBody,
    Stack,
    Text
  } from '@chakra-ui/react'
  
import { Amplify, Storage } from 'aws-amplify';

function Upload({url, title}) {
    const bg = useColorModeValue('blue.500', 'red.200')
    const color = useColorModeValue('white', 'gray.800')
    const [file, setFile] = useState();
    const [preview, setPreview] = useState();
    const [data, setData] = useState([]);
    const { isOpen, onOpen, onClose  } = useDisclosure() 

    function handleFile(e){        
        onClose()
        setFile(e.target.files[0])
        setPreview(URL.createObjectURL(e.target.files[0]))
        console.log(e.target.files[0])
    }

    function handleUpload(e){
        e.preventDefault()
        const formData = new FormData()
        formData.append('image', file)
        fetch(
            url,
            {
                method:'POST',
                body:formData
            }
        ).then(
                (response) => response.json()
            )
        .then(
            async (result) => {
                console.log('success', result)     
                Amplify.configure({
                    Auth: {
                      identityPoolId: 'eu-north-1:d5d60edb-8701-4c17-b9b6-8b8ca908072e', //REQUIRED - Amazon Cognito Identity Pool ID
                      region: 'eu-north-1', // REQUIRED - Amazon Cognito Region
                    },
                    Storage: {
                      AWSS3: {
                        bucket: 'fbm-storage', //REQUIRED -  Amazon S3 bucket name
                        region: 'eu-north-1', //OPTIONAL -  Amazon service region
                      }
                    }
                  });
                Storage.configure({ level: 'public' });

                for (const file of result.similar_index) {
                    let filename = file.id + ".jpg"
                    file["image"] = await Storage.get(filename, { 
                        level: 'public',validateObjectExistence: true 
                        })
                }

                setData(result.similar_index)
                onOpen()
            }
        )
        .catch(error => {
            console.error("Error:", error)
        })
    }

  return (
    <Center py={6}>
        
      <Box
        maxW={'1100px'}
        w={'1100px'}
        bg={useColorModeValue('#EBF8FF', 'gray.900')}
        boxShadow={'2xl'}
        rounded={'lg'}
        p={6}
        textAlign={'center'}>
            <Heading
        fontSize={{ base: "2rem", md: "3rem" }}
        marginBottom={{ base: "0.5rem", md: "1rem" }}
      >
        {title}
      </Heading>
        <SimpleGrid spacing={4} columns={2} minChildWidth='250px'>
            <Card  align={"center"} direction={"row"}  variant={"outline"} >
                <CardBody >
                    <Text>Choose a file</Text>
                    <form onSubmit={handleUpload}>
                        <Input type='file' onChange={handleFile}></Input>
                        <Button mt={3} colorScheme='blue' type='submit'>Submit</Button>
                    </form>                    
                </CardBody>
            </Card>
            <Card  variant={"outline"}>
                <CardBody align={"center"}>                    
                    <Heading fontSize={'2xl'} fontFamily={'body'}>Your Selection</Heading>
                    <Image align={"center"} src={preview} height={300}></Image>                    
                </CardBody>
            </Card>
        </SimpleGrid>
        <Collapse in={isOpen} animateOpacity>
            <Box
            p='40px'
            color='white'
            mt='4'            
            bg={useColorModeValue('white', 'gray.900')}
            rounded='md'
            shadow='md'
            >
             <Center>    
             <Stack
          marginTop="1rem"
          direction={{ base: "column", md: "row" }}
          gap="1rem"
          alignItems="stretch"
          justifyContent={"center"}
        >   
            {
                data.map((d) => {return(<ImageCard 
                    title = {d.id}
                    category = {d.category}
                    distance = {d.distances}
                    image={d.image}></ImageCard >)}
                )
            }
            </Stack>      
        </Center>
            </Box>
        </Collapse>
       
    </Box>
    </Center>
    
  );
}

export default Upload;