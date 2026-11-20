import { DurableObject } from "cloudflare:workers";

export class SPWSContainer extends DurableObject {
  async fetch(request) {
    const container = this.ctx.container;
    if (!container.running) {
      container.start();
    }

    const port = container.getTcpPort(8080);
    let lastError;
    for (let i = 0; i < 60; i++) {
      try {
        const health = await port.fetch("http://container/healthz");
        if (health.ok) {
          const url = new URL(request.url);
          url.protocol = "http:";
          url.host = "container";
          const forwarded = new Request(url, request);
          forwarded.headers.delete("host");
          return await port.fetch(forwarded);
        }
        lastError = new Error(`Health check returned ${health.status}`);
      } catch (err) {
        lastError = err;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return new Response(
      `SPWS container did not become ready: ${lastError}`,
      { status: 503 }
    );
  }
}

export default {
  async fetch(request, env) {
    return env.SPWS_APP.getByName("spws-main").fetch(request);
  },
};
