'use client'

import {
  Heading,
  Avatar,
  Box,
  Center,
  Text,
  Stack,
  Button,
  useColorModeValue,
} from '@chakra-ui/react'

import { FaLinkedin,FaGithub } from "react-icons/fa";
export default function SocialProfile() {
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
        <Avatar
          size={'xl'}
          src={
            'https://media.licdn.com/dms/image/C4E03AQG_ySMdOu7t3w/profile-displayphoto-shrink_800_800/0/1653494783852?e=1697673600&v=beta&t=js-o9qI3qC4hgE7mV3Q5uiz211QyetAiF4RpR_WuVV4'
          }
          mb={4}
          pos={'relative'}
          _after={{
            content: '""',
            w: 4,
            h: 4,
            bg: 'green.300',
            border: '2px solid white',
            rounded: 'full',
            pos: 'absolute',
            bottom: 0,
            right: 3,
          }}
        />
        <Heading fontSize={'2xl'} fontFamily={'body'}>
          Pavundeep Dhillon
        </Heading>
        <Text fontWeight={600} color={'gray.500'} mb={4}>
          @pddhillon
        </Text>
        <Text
          textAlign={'center'}
          color={useColorModeValue('gray.700', 'gray.400')}
          px={3}>
          4+ years as a Full Stack .NET Core developer
        </Text>
        <Stack mt={4} direction={'row'} spacing={2}>
        <Button 
            flex={1} rounded={'full'} colorScheme='linkedin' leftIcon={<FaLinkedin />}>
          <a href="https://www.linkedin.com/in/pavundeep-dhillon-86a505138/">Linkedin</a>
        </Button>
        <Button 
            flex={1} rounded={'full'} colorScheme='gray' leftIcon={<FaGithub />}>
          <a href="https://github.com/PDDhillon/facebook-marketplaces-recommendation-ranking-system">Github</a>
        </Button>
        </Stack>
      </Box>
    </Center>
  )
}