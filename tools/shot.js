
    const { chromium } = require('playwright');
    (async () => {
      const browser = await chromium.launch();
      const context = await browser.newContext({ deviceScaleFactor: 2 });
      const page = await context.newPage();
      await page.goto(process.argv[2], { waitUntil: 'networkidle' });
      await page.waitForTimeout(1500);
      await page.screenshot({ path: process.argv[3], fullPage: true, type: 'png' });
      await browser.close();
    })();
    