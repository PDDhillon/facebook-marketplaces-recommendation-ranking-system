import './App.css';
import { Button, ButtonGroup } from '@chakra-ui/react'
import { Input } from '@chakra-ui/react'
import { useState } from 'react';
import { Image } from '@chakra-ui/react'
import { Card, CardHeader, CardBody, CardFooter } from '@chakra-ui/react'
import { Center, Square, Circle } from '@chakra-ui/react'
import {
    List,
    ListItem,
    ListIcon,
    OrderedList,
    UnorderedList,
  } from '@chakra-ui/react'

function Upload({url, title}) {
    const [file, setFile] = useState();
    const [data, setData] = useState([]);

    function handleFile(e){
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
            }
        )
        .catch(error => {
            console.error("Error:", error)
        })
    }

  return (
    <div className="Upload">
      <header className="Upload-header">
        <h2>{title}</h2>
        <form onSubmit={handleUpload}>
            <Input type='file' onChange={handleFile}></Input>
            <Button colorScheme='blue' type='submit'>Submit</Button>
        </form>
        {/* <Card>
            <CardBody>
                <Center>                    
                    <Image borderRadius='lg' boxSize='250px' src={fileUrl}/>
                </Center>
            </CardBody>
        </Card> */}
        <Card>
            <CardBody>
                <Center>    
                    <UnorderedList>     
                    {
                        data.map((d) => {return(<ListItem>{d}</ListItem>)}
                        )
                    }
                    </UnorderedList>      
                </Center>
            </CardBody>
        </Card>
      </header>
    </div>
  );
}

export default Upload;