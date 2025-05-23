const { exec } = require('child_process');
const path = require('path');

module.exports = function (context, options) {
  return {
    name: 'fetch-substack-posts',
    async loadContent() {
      return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, '../scripts/fetchLatestPosts.js');
        exec(`node ${scriptPath}`, (error, stdout, stderr) => {
          if (error) {
            console.error('Error fetching Substack posts:', error);
            console.error('stderr:', stderr);
            // Don't fail the build, just log the error
            resolve(null);
          } else {
            console.log('Substack posts fetch output:', stdout);
            resolve(null);
          }
        });
      });
    },
  };
};
