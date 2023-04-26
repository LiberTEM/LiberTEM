import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import time
from fractions import Fraction

import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()


class VideoGenerateTrack(MediaStreamTrack):
    """
    A video stream track that generates some images.
    """

    kind = "video"

    def __init__(self):
        super().__init__()  # don't forget this!
        self.time_base = Fraction(1, 90000)
        self.start = None
        self.last = None
        self.counter = 0
        self.fps_cap = 60

    async def recv(self):
        now = time.monotonic()
        if self.start is None:
            pts = 0
            self.start = now
            self.last = now
        else:
            # Wait if we are going too fast
            while True:
                elapsed = now - self.last
                if elapsed < 1/self.fps_cap:
                    await asyncio.sleep(1/self.fps_cap - elapsed)
                    now = time.monotonic()
                else:
                    self.last = now
                    break
            pts = int((now - self.start) / self.time_base)
        # Random data with some systematically changing and static features 
        img = np.random.randint(low=0, high=255, size=(123, 456, 3), dtype=np.uint8)
        
        img[:60, :30] = 0
        img[:60, :60, 2] = np.uint8(int(((now - self.start)/8*255) % 255))
        img[:50, :50, 2] = np.uint8(int(((now - self.start)/4*255) % 255))
        img[:40, :40, 2] = np.uint8(int(((now - self.start)/2*255) % 255))
        
        img[:30, :30, 2] = np.uint8(int(((now - self.start)*255) % 255))
        img[:20, :20] = (255, 0, 0)
        img[:10, :10] = (0, 255, 0)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = self.time_base
        self.counter += 1
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html")).read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js")).read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(VideoGenerateTrack())
    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC NumPy array stream prototype"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
