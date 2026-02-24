#include "qt_utils/ProgressUtil.hpp"

#include <QProgressBar>
#include <QStatusBar>

#include <algorithm>
#include <cmath>

namespace {

/// Replace every occurrence of @p token in @p text with @p value.
auto replaceToken(QString text, const QString& token, const QString& value)
    -> QString
{
    if (text.contains(token)) {
        text.replace(token, value);
    }
    return text;
}

}  // namespace

namespace qt_utils {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

ProgressUtil::ProgressUtil(QStatusBar* statusBar, QObject* parent)
    : QObject(parent)
    , statusBar_(statusBar)
{
}

ProgressUtil::~ProgressUtil()
{
    stopProgress();

    if (animTimer_) {
        if (animTimer_->isActive()) {
            animTimer_->stop();
        }
        delete animTimer_;
    }
}

// ---------------------------------------------------------------------------
// Animated spinner
// ---------------------------------------------------------------------------

void ProgressUtil::startAnimation(const QString& message)
{
    animFrame_ = 0;
    animMessage_ = message;

    if (!animTimer_) {
        animTimer_ = new QTimer(this);
        connect(animTimer_, &QTimer::timeout, this, &ProgressUtil::onAnimationTick);
    }

    if (statusBar_) {
        statusBar_->showMessage(message + QStringLiteral(" |"), 0);
    }
    animTimer_->start(300);
}

void ProgressUtil::stopAnimation(const QString& message, int timeout)
{
    if (animTimer_ && animTimer_->isActive()) {
        animTimer_->stop();
    }

    if (statusBar_) {
        statusBar_->showMessage(message, timeout);
    }
}

// ---------------------------------------------------------------------------
// Progress bar management
// ---------------------------------------------------------------------------

void ProgressUtil::setProgressBar(QProgressBar* progressBar)
{
    if (progressBar_ == progressBar) {
        return;
    }

    // Restore previous bar before switching
    if (progressBar_ && progressBar_ != progressBar) {
        restoreProgressBarFont();
        restoreProgressBarFormat();
        progressBar_->setVisible(false);
    }

    progressBar_ = progressBar;

    if (progressBar_) {
        storedFormat_ = progressBar_->format();
        storedFont_ = progressBar_->font();
        storedMinimum_ = progressBar_->minimum();
        storedMaximum_ = progressBar_->maximum();
        hasStoredFont_ = true;
        hasStoredFormat_ = true;
        progressBar_->setVisible(false);
    }
}

void ProgressUtil::configureProgressBar(const ProgressBarOptions& options)
{
    defaultOptions_ = options;

    if (!progressActive_) {
        applyProgressBarFont();
        if (progressBar_) {
            if (defaultOptions_.textMode == ProgressTextMode::None) {
                progressBar_->setTextVisible(false);
            } else if (hasStoredFormat_) {
                progressBar_->setTextVisible(true);
                progressBar_->setFormat(storedFormat_);
            }
        }
    }
}

void ProgressUtil::startProgress(int totalSteps, const ProgressBarOptions* options)
{
    if (!progressBar_) {
        return;
    }

    progressActive_ = true;
    progressTotal_ = std::max(0, totalSteps);
    progressValue_ = 0;
    activeOptions_ = options ? *options : defaultOptions_;

    applyProgressBarFont();

    if (progressTotal_ <= 0) {
        progressBar_->setRange(0, 0);  // busy indicator
    } else {
        progressBar_->setRange(0, progressTotal_);
        progressBar_->setValue(0);
    }

    if (activeOptions_.textMode == ProgressTextMode::None) {
        progressBar_->setTextVisible(false);
    } else {
        progressBar_->setTextVisible(true);
    }

    updateProgressFormat();
    progressBar_->setVisible(true);
}

void ProgressUtil::updateProgress(int completedSteps)
{
    if (!progressBar_ || !progressActive_) {
        return;
    }

    if (progressTotal_ > 0) {
        progressValue_ = std::clamp(completedSteps, 0, progressTotal_);
    } else {
        progressValue_ = std::max(0, completedSteps);
    }
    progressBar_->setValue(progressValue_);

    updateProgressFormat();
}

void ProgressUtil::advanceProgress(int stepDelta)
{
    if (!progressActive_) {
        return;
    }

    int newValue = progressValue_ + stepDelta;
    if (progressTotal_ > 0) {
        newValue = std::clamp(newValue, 0, progressTotal_);
    } else {
        newValue = std::max(0, newValue);
    }
    updateProgress(newValue);
}

void ProgressUtil::stopProgress(bool resetValue)
{
    progressActive_ = false;
    progressTotal_ = 0;
    progressValue_ = 0;

    if (progressBar_) {
        if (resetValue) {
            progressBar_->setRange(storedMinimum_, storedMaximum_);
            progressBar_->setValue(storedMinimum_);
        }
        restoreProgressBarFont();
        restoreProgressBarFormat();
        progressBar_->setVisible(false);
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

auto ProgressUtil::isProgressActive() const -> bool { return progressActive_; }
auto ProgressUtil::progressTotal() const -> int { return progressTotal_; }
auto ProgressUtil::progressValue() const -> int { return progressValue_; }

// ---------------------------------------------------------------------------
// Private slots
// ---------------------------------------------------------------------------

void ProgressUtil::onAnimationTick()
{
    static const QChar kFrames[] = {'|', '/', '-', '\\'};
    animFrame_ = (animFrame_ + 1) % 4;
    if (statusBar_) {
        statusBar_->showMessage(
            animMessage_ + QStringLiteral(" ") + kFrames[animFrame_], 0);
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void ProgressUtil::applyProgressBarFont()
{
    if (!progressBar_) {
        return;
    }

    const auto& opts = progressActive_ ? activeOptions_ : defaultOptions_;
    if (opts.fontPointSize > 0) {
        QFont font = progressBar_->font();
        if (font.pointSize() != opts.fontPointSize) {
            font.setPointSize(opts.fontPointSize);
            progressBar_->setFont(font);
        }
    } else if (hasStoredFont_) {
        progressBar_->setFont(storedFont_);
    }
}

void ProgressUtil::restoreProgressBarFont()
{
    if (progressBar_ && hasStoredFont_) {
        progressBar_->setFont(storedFont_);
    }
}

void ProgressUtil::restoreProgressBarFormat()
{
    if (progressBar_) {
        if (hasStoredFormat_) {
            progressBar_->setFormat(storedFormat_);
        } else {
            progressBar_->setFormat(QStringLiteral("%p%"));
        }
        progressBar_->setTextVisible(true);
    }
}

void ProgressUtil::updateProgressFormat()
{
    if (!progressBar_) {
        return;
    }

    const auto& opts = progressActive_ ? activeOptions_ : defaultOptions_;

    if (opts.textMode == ProgressTextMode::None) {
        progressBar_->setTextVisible(false);
        return;
    }

    progressBar_->setTextVisible(true);

    // When not actively tracking, restore the original format
    if (!progressActive_) {
        if (hasStoredFormat_) {
            progressBar_->setFormat(storedFormat_);
        }
        return;
    }

    const int total = progressTotal_;
    const int maxVal = (total > 0) ? total : progressValue_;
    const int value = std::clamp(progressValue_, 0, maxVal);
    const int remaining = (total > 0) ? std::max(0, total - value) : 0;
    const int precision = std::max(0, opts.percentPrecision);
    const double percent =
        (total > 0)
            ? (static_cast<double>(value) * 100.0) / static_cast<double>(total)
            : 0.0;

    QString formatted;

    switch (opts.textMode) {
        case ProgressTextMode::Percent: {
            if (total <= 0) {
                formatted = QStringLiteral("--");
            } else if (precision == 0) {
                formatted = QString::number(
                    static_cast<int>(std::round(percent)));
            } else {
                formatted = QString::number(percent, 'f', precision);
            }
            formatted.append(QStringLiteral("%"));
            break;
        }
        case ProgressTextMode::Fraction: {
            if (total > 0) {
                formatted =
                    QStringLiteral("%1 / %2").arg(value).arg(total);
            } else {
                formatted = QString::number(value);
            }
            break;
        }
        case ProgressTextMode::DoneRemaining: {
            if (total > 0) {
                formatted = QStringLiteral("%1 done : %2 left")
                                .arg(value)
                                .arg(remaining);
            } else {
                formatted = QStringLiteral("%1 done").arg(value);
            }
            break;
        }
        case ProgressTextMode::Custom: {
            formatted = opts.customFormat;
            formatted = replaceToken(
                formatted, QStringLiteral("{value}"),
                QString::number(value));
            formatted = replaceToken(
                formatted, QStringLiteral("{total}"),
                total > 0 ? QString::number(total) : QStringLiteral("-"));
            formatted = replaceToken(
                formatted, QStringLiteral("{remaining}"),
                QString::number(remaining));
            if (precision == 0) {
                formatted = replaceToken(
                    formatted, QStringLiteral("{percent}"),
                    QString::number(
                        static_cast<int>(std::round(percent))));
            } else {
                formatted = replaceToken(
                    formatted, QStringLiteral("{percent}"),
                    QString::number(percent, 'f', precision));
            }
            break;
        }
        case ProgressTextMode::None:
            break;  // Already handled above
    }

    if (!opts.prefix.isEmpty()) {
        formatted.prepend(opts.prefix);
    }
    if (!opts.suffix.isEmpty()) {
        formatted.append(opts.suffix);
    }

    if (!formatted.isEmpty()) {
        progressBar_->setFormat(formatted);
    }
}

}  // namespace qt_utils
