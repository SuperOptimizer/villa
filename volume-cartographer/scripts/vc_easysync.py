#!/usr/bin/env python3
"""
VC EasySync - Simple S3 Upload Tool with Overwrite Protection

A stripped-down S3 sync tool focused on safe uploads without database tracking.
Maps identifiers (e.g., "scroll1", "scroll5") to specific S3 buckets.

Usage:
    python vc_easysync.py upload <file_or_folder> <identifier> [--profile=<aws_profile>] [--dry-run]
    python vc_easysync.py check <file_or_folder> <identifier> [--profile=<aws_profile>]
    python vc_easysync.py test-bucket <identifier> [--profile=<aws_profile>]
    python vc_easysync.py list-ids

Examples:
    # Test bucket connection
    python vc_easysync.py test-bucket scroll5

    # Check what would be uploaded
    python vc_easysync.py check ./segments scroll5

    # Upload with dry-run
    python vc_easysync.py upload ./segments scroll5 --dry-run

    # Actually upload
    python vc_easysync.py upload ./segments scroll5
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from enum import Enum


class UploadStatus(Enum):
    NEW = "new"           # File doesn't exist on S3
    EXISTS = "exists"     # File exists on S3 (would overwrite)
    SKIP = "skip"        # File should be skipped


class VCEasySync:
    # Mapping of identifiers to S3 configurations
    BUCKET_MAPPINGS = {
        "scroll5": {
            "bucket": "vesuvius-challenge",
            "prefix": "PHerc0172"
        },
    }

    def __init__(self, identifier, aws_profile=None):
        """Initialize with an identifier that maps to S3 configuration"""
        if identifier not in self.BUCKET_MAPPINGS:
            available = ", ".join(sorted(self.BUCKET_MAPPINGS.keys()))
            raise ValueError(f"Unknown identifier '{identifier}'. Available: {available}")

        config = self.BUCKET_MAPPINGS[identifier]
        self.identifier = identifier
        self.s3_bucket = config["bucket"]
        self.s3_prefix = config["prefix"].rstrip('/')
        self.aws_profile = aws_profile

    @classmethod
    def list_identifiers(cls):
        """List all available identifiers and their S3 mappings"""
        print("Available identifiers and their S3 mappings:\n")
        for identifier, config in sorted(cls.BUCKET_MAPPINGS.items()):
            print(f"  {identifier:10} → s3://{config['bucket']}/{config['prefix']}/")

    def _run_aws_command(self, cmd):
        """Run AWS CLI command with optional profile"""
        if self.aws_profile:
            cmd.extend(['--profile', self.aws_profile])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result
        except subprocess.CalledProcessError as e:
            # Check for common authentication errors
            if "Unable to locate credentials" in str(e.stderr):
                print("\n❌ AWS credentials not found!")
                print("   Please configure AWS credentials using one of these methods:")
                print("   1. Run: aws configure")
                print("   2. Use --profile flag with an existing profile")
                print("   3. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
                print("\nTo see available profiles, run: python vc_easysync.py list-profiles")
                sys.exit(1)
            elif "ExpiredToken" in str(e.stderr) or "TokenRefreshRequired" in str(e.stderr):
                print("\n❌ AWS credentials have expired!")
                if self.aws_profile:
                    print(f"   Profile '{self.aws_profile}' needs to be refreshed.")
                print("   Please refresh your AWS credentials and try again.")
                sys.exit(1)
            else:
                # Re-raise the original error for other cases
                raise

    def _should_skip_file(self, filepath, relative_path):
        """Check if a file should be skipped based on ignore rules"""
        filename = os.path.basename(filepath)

        # Skip hidden files
        if filename.startswith('.'):
            return True

        # Skip .obj files
        if filename.endswith('.obj'):
            return True

        # Skip files in directories containing 'layers'
        path_parts = relative_path.split(os.sep)
        if any('layers' in part.lower() for part in path_parts[:-1]):
            return True

        # Skip hidden directories
        if any(part.startswith('.') for part in path_parts[:-1]):
            return True

        return False

    def scan_local_files(self, path):
        """Scan local file or directory for files to upload"""
        files = {}

        if os.path.isfile(path):
            # Single file
            if not self._should_skip_file(path, os.path.basename(path)):
                stat = os.stat(path)
                files[os.path.basename(path)] = {
                    'path': path,
                    'relative_path': os.path.basename(path),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                }
        else:
            # Directory
            base_dir = os.path.abspath(path)
            for root, dirs, filenames in os.walk(base_dir):
                # Skip hidden directories and directories containing 'layers'
                dirs[:] = [d for d in dirs if not d.startswith('.') and 'layers' not in d.lower()]

                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    relative_path = os.path.relpath(filepath, base_dir)

                    if not self._should_skip_file(filepath, relative_path):
                        stat = os.stat(filepath)
                        files[relative_path] = {
                            'path': filepath,
                            'relative_path': relative_path,
                            'size': stat.st_size,
                            'mtime': stat.st_mtime
                        }

        return files

    def check_s3_files(self, file_paths):
        """Check which files exist on S3"""
        existing_files = {}

        print(f"Checking S3 bucket: s3://{self.s3_bucket}/{self.s3_prefix}/")

        for relative_path in file_paths:
            s3_key = f"{self.s3_prefix}/{relative_path}"

            # Use head-object to check if file exists
            cmd = [
                'aws', 's3api', 'head-object',
                '--bucket', self.s3_bucket,
                '--key', s3_key
            ]

            try:
                result = self._run_aws_command(cmd)
                data = json.loads(result.stdout)

                existing_files[relative_path] = {
                    'size': data.get('ContentLength', 0),
                    'mtime': data.get('LastModified', ''),
                    'etag': data.get('ETag', '').strip('"')
                }
            except subprocess.CalledProcessError:
                # File doesn't exist on S3
                pass

        return existing_files

    def analyze_uploads(self, local_files):
        """Analyze what would be uploaded and what would be overwritten"""
        print("\nAnalyzing files...")

        # Check S3 for existing files
        s3_files = self.check_s3_files(local_files.keys())

        upload_plan = {}

        for relative_path, local_info in local_files.items():
            if relative_path in s3_files:
                s3_info = s3_files[relative_path]

                # File exists on S3
                size_diff = local_info['size'] != s3_info['size']

                upload_plan[relative_path] = {
                    'status': UploadStatus.EXISTS,
                    'local_size': local_info['size'],
                    's3_size': s3_info['size'],
                    'size_diff': size_diff,
                    'local_path': local_info['path']
                }
            else:
                # New file
                upload_plan[relative_path] = {
                    'status': UploadStatus.NEW,
                    'local_size': local_info['size'],
                    'local_path': local_info['path']
                }

        return upload_plan

    def format_size(self, size):
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def show_upload_plan(self, upload_plan):
        """Display the upload plan to the user"""
        new_files = []
        overwrites = []

        for path, info in sorted(upload_plan.items()):
            if info['status'] == UploadStatus.NEW:
                new_files.append((path, info))
            elif info['status'] == UploadStatus.EXISTS:
                overwrites.append((path, info))

        print(f"\nUpload Summary for '{self.identifier}':")
        print(f"  Target: s3://{self.s3_bucket}/{self.s3_prefix}/")
        print(f"  New files: {len(new_files)}")
        print(f"  Overwrites: {len(overwrites)}")

        if new_files:
            print(f"\n📤 New files to upload ({len(new_files)}):")
            total_size = 0
            for path, info in new_files[:10]:  # Show first 10
                size_str = self.format_size(info['local_size'])
                print(f"    {path} ({size_str})")
                total_size += info['local_size']

            if len(new_files) > 10:
                print(f"    ... and {len(new_files) - 10} more files")

            print(f"  Total size: {self.format_size(total_size)}")

        if overwrites:
            print(f"\n⚠️  Files that would be OVERWRITTEN ({len(overwrites)}):")
            for path, info in overwrites[:10]:  # Show first 10
                local_size = self.format_size(info['local_size'])
                s3_size = self.format_size(info['s3_size'])

                if info['size_diff']:
                    print(f"    {path}")
                    print(f"      Local: {local_size}, S3: {s3_size} (SIZE DIFFERS)")
                else:
                    print(f"    {path} (same size: {local_size})")

            if len(overwrites) > 10:
                print(f"    ... and {len(overwrites) - 10} more files")

        return len(new_files), len(overwrites)

    def perform_upload(self, relative_path, local_path):
        """Upload a single file to S3"""
        s3_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/{relative_path}"

        cmd = ['aws', 's3', 'cp', local_path, s3_path]
        self._run_aws_command(cmd)

        return True

    def upload(self, path, dry_run=False):
        """Main upload function"""
        # Scan local files
        print(f"Scanning local path: {path}")
        local_files = self.scan_local_files(path)

        if not local_files:
            print("No files to upload (all files filtered by ignore rules)")
            return

        print(f"Found {len(local_files)} files to process")

        # Analyze what would be uploaded
        upload_plan = self.analyze_uploads(local_files)

        # Show the plan
        new_count, overwrite_count = self.show_upload_plan(upload_plan)

        if dry_run:
            print("\n--dry-run mode: No files will be uploaded")
            return

        if new_count == 0 and overwrite_count == 0:
            print("\n✓ Nothing to upload!")
            return

        # Get confirmation
        print("\n" + "="*50)
        if overwrite_count > 0:
            print(f"⚠️  WARNING: {overwrite_count} files will be OVERWRITTEN on S3!")
            print("This action cannot be undone.")

        print(f"\nTotal files to upload: {new_count + overwrite_count}")
        response = input("\nProceed with upload? [y/N]: ").strip().lower()

        if response != 'y':
            print("Upload cancelled.")
            return

        # Perform uploads
        print("\nUploading files...")
        success_count = 0
        failed_files = []

        for i, (relative_path, info) in enumerate(sorted(upload_plan.items()), 1):
            try:
                # Show progress for large uploads
                if len(upload_plan) > 10 and i % 10 == 0:
                    print(f"  Progress: {i}/{len(upload_plan)} files...")
                elif len(upload_plan) <= 10:
                    status = "Overwriting" if info['status'] == UploadStatus.EXISTS else "Uploading"
                    print(f"  {status}: {relative_path}")

                self.perform_upload(relative_path, info['local_path'])
                success_count += 1

            except Exception as e:
                print(f"  ❌ Failed: {relative_path} - {str(e)}")
                failed_files.append(relative_path)

        # Final summary
        print(f"\n{'='*50}")
        print(f"✓ Upload complete: {success_count}/{len(upload_plan)} files successful")

        if failed_files:
            print(f"\n❌ Failed uploads ({len(failed_files)}):")
            for path in failed_files[:10]:
                print(f"  - {path}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")

    def check_status(self, path):
        """Check upload status without uploading"""
        # Scan local files
        print(f"Scanning local path: {path}")
        local_files = self.scan_local_files(path)

        if not local_files:
            print("No files found (all files filtered by ignore rules)")
            return

        print(f"Found {len(local_files)} files to check")

        # Analyze what would be uploaded
        upload_plan = self.analyze_uploads(local_files)

        # Show the plan
        self.show_upload_plan(upload_plan)


def main():
    parser = argparse.ArgumentParser(
        description='VC EasySync - Simple S3 upload tool with overwrite protection'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload files to S3')
    upload_parser.add_argument('path', help='File or directory to upload')
    upload_parser.add_argument('identifier', help='Target identifier (e.g., scroll1, scroll5)')
    upload_parser.add_argument('--profile', help='AWS profile to use')
    upload_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be uploaded without doing it')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check upload status without uploading')
    check_parser.add_argument('path', help='File or directory to check')
    check_parser.add_argument('identifier', help='Target identifier (e.g., scroll1, scroll5)')
    check_parser.add_argument('--profile', help='AWS profile to use')

    # List identifiers command
    list_parser = subparsers.add_parser('list-ids', help='List available identifiers')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'list-ids':
            VCEasySync.list_identifiers()

        elif args.command == 'upload':
            if not os.path.exists(args.path):
                print(f"Error: Path does not exist: {args.path}")
                sys.exit(1)

            syncer = VCEasySync(args.identifier, args.profile)
            syncer.upload(args.path, args.dry_run)

        elif args.command == 'check':
            if not os.path.exists(args.path):
                print(f"Error: Path does not exist: {args.path}")
                sys.exit(1)

            syncer = VCEasySync(args.identifier, args.profile)
            syncer.check_status(args.path)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()