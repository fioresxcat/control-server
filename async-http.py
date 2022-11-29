import aiohttp
import asyncio
import time



def object_detect_loop():
    async def main():
        async with aiohttp.ClientSession() as session:

            cam1_url = 'http://localhost:8001/cam1'
            async with session.post(cam1_url, json={
                'data': [
                    {
                    'time': time.time(),
                    'feature': [1, 2, 3, 4],
                    'id': 1
                    },
                    {
                    'time': time.time(),
                    'feature': [5, 6, 7, 8],
                    'id': 2
                    }
                ]
            }) as resp:
                result = await resp.json()
                print(result)
                
    asyncio.run(main())

if __name__ == '__main__':
    object_detect_loop()