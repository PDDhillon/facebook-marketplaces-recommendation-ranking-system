import './App.css';import {
    Card,
    CardBody,
    Image,
    Text,
    Divider,
    Heading,
    Stack,
  } from '@chakra-ui/react'



  
function ImageCard({title,image, distance, category}) {
  return (
    <div className="ImageCard">
        <Card size="sm">
            <CardBody>
                <Image
                src={image}
                boxSize='150px'
                objectFit='cover'
                borderRadius='lg'
                />
                <Stack mt='6' spacing='3'>
                <Heading size='sm'>{title}</Heading>
                
                <Heading size='sm'>Category: {category}</Heading>
                <Text color='blue.600' >
                    Distance: {distance}
                </Text>
                </Stack>
            </CardBody>
            <Divider />
        </Card>
    </div>
  );
}

export default ImageCard;





