import './App.css';
import { useState } from 'react';
import {
    Box,
    useColorModeValue,
    ListItem,
    UnorderedList,
    useDisclosure,
    Collapse,
    Heading,
    Stack,
    Center,
    Button,
    Input
  } from '@chakra-ui/react'

function Upload({url, title}) {
    const [file, setFile] = useState();
    const [data, setData] = useState([]);
    const { isOpen, onOpen, onClose  } = useDisclosure()

    function handleFile(e){        
        onClose()
        setFile(e.target.files[0])
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
            (result) => {
                console.log('success', result)                
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
        maxW={'320px'}
        w={'full'}
        bg={useColorModeValue('white', 'gray.900')}
        boxShadow={'2xl'}
        rounded={'lg'}
        p={6}
        textAlign={'center'}>
        <Heading fontSize={'2xl'} fontFamily={'body'}>{title}</Heading>
        <Stack mt={5} direction='row'>
            <form onSubmit={handleUpload}>
                <Input type='file' onChange={handleFile}></Input>
                <Button mt={3} colorScheme='blue' type='submit'>Submit</Button>
            </form>
        </Stack>
        <Collapse in={isOpen} animateOpacity>
            <Box
            p='40px'
            color='white'
            mt='4'
            bg='teal.500'
            rounded='md'
            shadow='md'
            >
             <Center>    
            <UnorderedList>     
            {
                data.map((d) => {return(<ListItem>{d}</ListItem>)}
                )
            }
            </UnorderedList>      
        </Center>
            </Box>
        </Collapse>
       
    </Box>
    </Center>
  );
}

export default Upload;